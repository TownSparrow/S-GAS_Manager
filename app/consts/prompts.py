PROMPT_WITH_CONTEXT = (
    "Context from the knowledge base:\n{context}\n\n"
    "User's request: {message}\n\n"
    "Instructions: Analyze the context and give a brief, precise, full and specific answer "
    "based on the fragments provided. If the context lacks information, state so and respond "
    "briefly using general knowledge. Use the same language as the user uses.\n\n"
    "Response:"
)

PROMPT_WITHOUT_CONTEXT = (
    "User's request: {message}\n\n"
    "Give a short and precise answer. Use the same language as the user uses.\n\n"
    "Response:"
)

VLLM_STOP_SEQUENCES = ["\n\nRequest:", "\n\nUser:", "\n\nAssistant:"]
