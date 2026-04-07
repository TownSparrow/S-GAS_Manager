PROMPT_WITH_CONTEXT = (
    "Context from the knowledge base (fragments are ranked by relevance, "
    "with the most relevant appearing first):\n{context}\n\n"
    "User's request: {message}\n\n"
    "Instructions:\n"
    "1. The context contains related facts retrieved from the knowledge base along with "
    "their connections. Use these facts to construct a coherent and complete answer.\n"
    "2. Fragments may contain contradictory information. When conflicts arise, prioritize "
    "the fragments that appear earlier (they are more relevant to the query) and clearly "
    "state the most supported conclusion.\n"
    "3. Synthesize information across multiple fragments when they complement each other "
    "to provide a fuller answer.\n"
    "4. Give a brief, precise, full and specific answer based on the fragments provided. "
    "If the context lacks sufficient information, state so and respond briefly using "
    "general knowledge.\n"
    "5. Use the same language as the user uses.\n\n"
    "Response:"
)

PROMPT_WITHOUT_CONTEXT = (
    "User's request: {message}\n\n"
    "Give a short and precise answer. Use the same language as the user uses.\n\n"
    "Response:"
)

VLLM_STOP_SEQUENCES = ["\n\nRequest:", "\n\nUser:", "\n\nAssistant:"]
