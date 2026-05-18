#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BATCH_STARTED_AT=""
BATCH_DIR=""
SUMMARY_FILE=""

DEFAULT_CONFIGS=(
  "cfg/Qwen__Qwen2.5-7B-Instruct-AWQ__stress.json"
  "cfg/Qwen__Qwen2.5-7B-Instruct-AWQ__optimum.json"
  "cfg/Qwen__Qwen2.5-7B-Instruct-AWQ__max.json"
  "cfg/google__gemma-4-E2B-it__stress.json"
  "cfg/google__gemma-4-E2B-it__optimum.json"
  "cfg/google__gemma-4-E2B-it__max.json"
)

CONFIGS=("${DEFAULT_CONFIGS[@]}")
SCENARIOS=()
CONTINUE_ON_ERROR=1
VLLM_PORT=8000
FORCE_FREE_PORTS=${S_GAS_BENCH_FORCE_PORTS:-1}

usage() {
  cat <<'EOF'
Usage:
  scripts/run_full_benchmark_cycle.sh [options]

Runs every selected vLLM preset one by one, starts S-GAS API, executes every
selected benchmark scenario through all algorithm modes, and stores CSV/JSON/DOCX
artifacts in logs/benchmarks.

Options:
  --configs <list>       Comma-separated config JSON paths. Default: all six presets.
  --scenarios <list>     Comma-separated scenario names. Default: all tests/scenarios/*.json.
  --stop-on-error        Stop the whole cycle after the first failed preset/scenario.
  --continue-on-error    Continue after failures. Default.
  --no-force-ports       Do not kill existing processes on vLLM/API ports.
  -h, --help             Show this help.

Environment:
  S_GAS_BENCH_FORCE_PORTS=0  Same as --no-force-ports.
EOF
}

split_csv_to_configs() {
  local raw="$1"
  local item
  CONFIGS=()
  IFS=',' read -ra parts <<< "$raw"
  for item in "${parts[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [ -n "$item" ]; then
      CONFIGS+=("$item")
    fi
  done
}

split_csv_to_scenarios() {
  local raw="$1"
  local item
  SCENARIOS=()
  IFS=',' read -ra parts <<< "$raw"
  for item in "${parts[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [ -n "$item" ]; then
      SCENARIOS+=("$item")
    fi
  done
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --configs)
      shift
      [ "$#" -gt 0 ] || { echo "Missing value for --configs"; exit 2; }
      split_csv_to_configs "$1"
      ;;
    --scenarios)
      shift
      [ "$#" -gt 0 ] || { echo "Missing value for --scenarios"; exit 2; }
      split_csv_to_scenarios "$1"
      ;;
    --stop-on-error)
      CONTINUE_ON_ERROR=0
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      ;;
    --no-force-ports)
      FORCE_FREE_PORTS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
  shift
done

if [ ! -d "S-GAS_Manager_env" ]; then
  echo "Virtual environment is not created. Run scripts/install_env.sh first."
  exit 1
fi

source S-GAS_Manager_env/bin/activate

for tool in jq curl; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required command is missing: $tool"
    exit 1
  fi
done

if [ "${#SCENARIOS[@]}" -eq 0 ]; then
  while IFS= read -r scenario_file; do
    SCENARIOS+=("$(basename "$scenario_file" .json)")
  done < <(find tests/scenarios -maxdepth 1 -type f -name '*.json' | sort)
fi

if [ "${#SCENARIOS[@]}" -eq 0 ]; then
  echo "No scenarios found in tests/scenarios."
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  if [ ! -f "$config" ]; then
    echo "Config file is not found: $config"
    exit 1
  fi
done

BATCH_STARTED_AT="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="logs/benchmark_batch/$BATCH_STARTED_AT"
mkdir -p "$BATCH_DIR"
SUMMARY_FILE="$BATCH_DIR/batch_summary.tsv"
printf 'preset\tscenario\tstatus\tcomparison_json\tdocx_report\tresponse_json\n' >"$SUMMARY_FILE"

export HF_HUB_CACHE="$ROOT_DIR/models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VLLM_PID=""
API_PID=""
VLLM_CMD=()

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

jq_value() {
  local config="$1"
  local filter="$2"
  jq -r "$filter // empty" "$config"
}

jq_json() {
  local config="$1"
  local filter="$2"
  jq -c "$filter // empty" "$config"
}

port_in_use() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser "${port}/tcp" >/dev/null 2>&1
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    curl -s "http://127.0.0.1:${port}" >/dev/null 2>&1
  fi
}

free_port() {
  local port="$1"
  if ! port_in_use "$port"; then
    return 0
  fi

  if [ "$FORCE_FREE_PORTS" != "1" ]; then
    echo "Port $port is already in use. Stop the process or rerun without --no-force-ports."
    exit 1
  fi

  log "Port $port is busy; stopping existing process on this port."
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  elif command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "$pids" ]; then
      kill $pids 2>/dev/null || true
    fi
  fi
  sleep 2
}

stop_servers() {
  set +e
  if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" 2>/dev/null; then
    log "Stopping S-GAS API (PID $API_PID)."
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    log "Stopping vLLM (PID $VLLM_PID)."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  API_PID=""
  VLLM_PID=""
  set -e
}

trap stop_servers EXIT INT TERM

wait_for_url() {
  local label="$1"
  local url="$2"
  local timeout_seconds="$3"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout_seconds" ]; then
      echo "$label did not become ready after ${timeout_seconds}s: $url"
      return 1
    fi
    sleep 3
  done
}

flag_supported() {
  local help_text="$1"
  local flag="$2"
  grep -q -- "$flag" <<< "$help_text"
}

append_value_arg() {
  local flag="$1"
  local value="$2"
  if [ -n "$value" ] && [ "$value" != "null" ]; then
    VLLM_CMD+=("$flag" "$value")
  fi
}

append_positive_value_arg() {
  local flag="$1"
  local value="$2"
  if [ -n "$value" ] && [ "$value" != "null" ] && [ "$value" != "0" ] && [ "$value" != "0.0" ]; then
    VLLM_CMD+=("$flag" "$value")
  fi
}

append_optional_value_arg() {
  local help_text="$1"
  local flag="$2"
  local value="$3"
  if [ -n "$value" ] && [ "$value" != "null" ]; then
    if flag_supported "$help_text" "$flag"; then
      VLLM_CMD+=("$flag" "$value")
    else
      log "Warning: installed vLLM does not support $flag; skipping it."
    fi
  fi
}

append_optional_positive_value_arg() {
  local help_text="$1"
  local flag="$2"
  local value="$3"
  if [ -n "$value" ] && [ "$value" != "null" ] && [ "$value" != "0" ] && [ "$value" != "0.0" ]; then
    if flag_supported "$help_text" "$flag"; then
      VLLM_CMD+=("$flag" "$value")
    else
      log "Warning: installed vLLM does not support $flag; skipping it."
    fi
  fi
}

build_vllm_command() {
  local config="$1"
  local vllm_help="$2"

  local model gpu_mem max_len max_seqs max_batched_tokens swap_space cpu_offload_gb
  local kv_cache_dtype kv_offloading_size quant dtype limit_mm_per_prompt prefix_caching

  model="$(jq_value "$config" '.vllm.model_name')"
  gpu_mem="$(jq_value "$config" '.vllm.gpu_memory_utilization')"
  max_len="$(jq_value "$config" '.vllm.max_model_len')"
  max_seqs="$(jq_value "$config" '.vllm.max_num_seqs')"
  max_batched_tokens="$(jq_value "$config" '.vllm.max_num_batched_tokens')"
  swap_space="$(jq_value "$config" '.vllm.swap_space')"
  cpu_offload_gb="$(jq_value "$config" '.vllm.cpu_offload_gb')"
  kv_cache_dtype="$(jq_value "$config" '.vllm.kv_cache_dtype')"
  kv_offloading_size="$(jq_value "$config" '.vllm.kv_offloading_size')"
  quant="$(jq_value "$config" '.vllm.quantization')"
  dtype="$(jq_value "$config" '.vllm.dtype')"
  limit_mm_per_prompt="$(jq_json "$config" '.vllm.limit_mm_per_prompt')"
  prefix_caching="$(jq_value "$config" '.vllm.enable_prefix_caching')"

  dtype=${dtype:-"auto"}
  VLLM_CMD=(vllm serve "$model" --host 0.0.0.0 --port "$VLLM_PORT")

  append_value_arg "--gpu-memory-utilization" "$gpu_mem"
  append_value_arg "--max-model-len" "$max_len"
  append_value_arg "--max-num-seqs" "$max_seqs"
  append_value_arg "--max-num-batched-tokens" "$max_batched_tokens"
  append_value_arg "--swap-space" "$swap_space"
  append_positive_value_arg "--cpu-offload-gb" "$cpu_offload_gb"
  append_value_arg "--kv-cache-dtype" "$kv_cache_dtype"
  append_optional_positive_value_arg "$vllm_help" "--kv-offloading-size" "$kv_offloading_size"
  append_value_arg "--dtype" "$dtype"

  if [ -n "$quant" ] && [ "$quant" != "null" ] && [ "$quant" != "none" ]; then
    VLLM_CMD+=("--quantization" "$quant")
  fi

  if [ "$(jq_value "$config" '.vllm.trust_remote_code')" = "true" ]; then
    VLLM_CMD+=("--trust-remote-code")
  fi

  if [ "$(jq_value "$config" '.vllm.language_model_only')" = "true" ]; then
    if flag_supported "$vllm_help" "--language-model-only"; then
      VLLM_CMD+=("--language-model-only")
    else
      log "Warning: installed vLLM does not support --language-model-only; using other text-only limits."
    fi
  fi

  if [ -n "$limit_mm_per_prompt" ] && [ "$limit_mm_per_prompt" != "null" ]; then
    append_optional_value_arg "$vllm_help" "--limit-mm-per-prompt" "$limit_mm_per_prompt"
  fi

  if [ "$prefix_caching" = "true" ]; then
    VLLM_CMD+=("--enable-prefix-caching")
  elif [ "$prefix_caching" = "false" ] && flag_supported "$vllm_help" "--no-enable-prefix-caching"; then
    VLLM_CMD+=("--no-enable-prefix-caching")
  fi
}

run_one_preset() {
  local config="$1"
  local preset_index="$2"
  local preset_count="$3"
  local config_abs="$ROOT_DIR/$config"
  local config_base model api_host api_port api_bind_host api_base profile_description
  local model_cache preset_dir vllm_log api_log response_file vllm_help

  config_base="$(basename "$config" .json)"
  model="$(jq_value "$config" '.vllm.model_name')"
  api_bind_host="$(jq_value "$config" '.api.host')"
  api_port="$(jq_value "$config" '.api.port')"
  profile_description="$(jq_value "$config" '.profile.description')"
  api_bind_host=${api_bind_host:-"0.0.0.0"}
  api_port=${api_port:-8080}
  api_host="127.0.0.1"
  api_base="http://${api_host}:${api_port}"
  preset_dir="$BATCH_DIR/$config_base"
  mkdir -p "$preset_dir"

  log "Main iteration: $config_base (${preset_index}/${preset_count})"
  log "Model: $model"
  if [ -n "$profile_description" ]; then
    log "Preset note: $profile_description"
  fi

  if [[ "$model" == google/gemma-4-E2B-it* ]]; then
    log "Gemma note: low-VRAM modes are experimental; failures will be recorded clearly."
  fi

  model_cache="models/models--$(echo "$model" | sed 's|/|--|g')"
  if [ ! -d "$model_cache" ]; then
    echo "Model is not downloaded locally: $model_cache"
    echo "Run scripts/install_env.sh first, or download $model into ./models."
    return 1
  fi

  export S_GAS_CONFIG_PATH="$config_abs"
  free_port "$VLLM_PORT"
  free_port "$api_port"

  vllm_help="$(vllm serve --help 2>&1 || true)"
  build_vllm_command "$config" "$vllm_help"

  vllm_log="$preset_dir/vllm.log"
  api_log="$preset_dir/api.log"

  printf '[%s] vLLM command:' "$(timestamp)" | tee "$preset_dir/vllm_command.txt"
  printf ' %q' "${VLLM_CMD[@]}" | tee -a "$preset_dir/vllm_command.txt"
  printf '\n' | tee -a "$preset_dir/vllm_command.txt"

  log "Starting vLLM for $config_base. Log: $vllm_log"
  "${VLLM_CMD[@]}" >"$vllm_log" 2>&1 &
  VLLM_PID=$!

  if ! wait_for_url "vLLM" "http://127.0.0.1:${VLLM_PORT}/v1/models" 900; then
    log "vLLM failed to become ready for $config_base. See $vllm_log."
    return 1
  fi

  log "Starting S-GAS API for $config_base. Log: $api_log"
  (
    cd "$ROOT_DIR"
    source S-GAS_Manager_env/bin/activate
    export S_GAS_CONFIG_PATH="$config_abs"
    unset HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    uvicorn run:app --host "$api_bind_host" --port "$api_port"
  ) >"$api_log" 2>&1 &
  API_PID=$!

  if ! wait_for_url "S-GAS API" "${api_base}/health" 180; then
    log "S-GAS API failed to become ready for $config_base. See $api_log."
    return 1
  fi

  for scenario_index in "${!SCENARIOS[@]}"; do
    local scenario="${SCENARIOS[$scenario_index]}"
    local human_scenario scenario_number scenario_count
    scenario_number=$((scenario_index + 1))
    scenario_count=${#SCENARIOS[@]}
    human_scenario="${scenario//_/ }"
    response_file="$preset_dir/${scenario}_response.json"

    log "Running scenario: $human_scenario (${scenario_number}/${scenario_count}) on $config_base"
    log "Algorithm order inside scenario: baseline -> hybrid_rag -> sgas_no_filtering -> sgas"

    if curl -fsS -X POST "${api_base}/api/benchmark/run/${scenario}" -o "$response_file"; then
      local status comparison docx
      status="$(jq -r '.status // "unknown"' "$response_file" 2>/dev/null || echo "unknown")"
      comparison="$(jq -r '.files.comparison // empty' "$response_file" 2>/dev/null || true)"
      docx="$(jq -r '.files.docx_report // .report_file // empty' "$response_file" 2>/dev/null || true)"
      log "Scenario completed: $scenario (status: $status)"
      if [ -n "$comparison" ]; then
        log "Comparison JSON: $comparison"
      fi
      if [ -n "$docx" ]; then
        log "DOCX report: $docx"
      fi
      log "CSV files are listed in the response: $response_file"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$config_base" "$scenario" "$status" "$comparison" "$docx" "$response_file" >>"$SUMMARY_FILE"
    else
      log "Scenario failed: $scenario on $config_base. See $response_file, $api_log, and $vllm_log."
      printf '%s\t%s\tfailed\t\t\t%s\n' "$config_base" "$scenario" "$response_file" >>"$SUMMARY_FILE"
      if [ "$CONTINUE_ON_ERROR" != "1" ]; then
        return 1
      fi
    fi
  done

  stop_servers
  log "Finished main iteration: $config_base (${preset_index}/${preset_count})"
}

log "S-GAS full benchmark cycle started."
log "Batch logs: $BATCH_DIR"
log "Presets: ${#CONFIGS[@]}"
log "Scenarios: ${SCENARIOS[*]}"
log "Generated CSV/JSON/DOCX benchmark artifacts will be stored in logs/benchmarks."

failed=0
for config_index in "${!CONFIGS[@]}"; do
  if ! run_one_preset "${CONFIGS[$config_index]}" "$((config_index + 1))" "${#CONFIGS[@]}"; then
    failed=1
    log "Preset failed: ${CONFIGS[$config_index]}"
    printf '%s\t-\tpreset_failed\t\t\t-\n' "$(basename "${CONFIGS[$config_index]}" .json)" >>"$SUMMARY_FILE"
    stop_servers
    if [ "$CONTINUE_ON_ERROR" != "1" ]; then
      exit 1
    fi
  fi
done

if [ "$failed" = "1" ]; then
  log "Benchmark cycle finished with failures. Check logs under $BATCH_DIR."
  exit 1
fi

log "Benchmark cycle completed successfully."
log "Batch logs: $BATCH_DIR"
log "Benchmark artifacts: logs/benchmarks"
