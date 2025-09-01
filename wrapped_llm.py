# wrapped_llm.py

class WrappedLLM:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt):
        """Takes a prompt and returns the response from the underlying LLM."""
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"[Error invoking LLM] {str(e)}"
