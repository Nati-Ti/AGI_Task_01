from together import *

class TogetherAIProcessor:
    def __init__(self, api_key):
        self.client = Together(api_key=api_key)
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
        You will be given texts related to a certain topic. Write a summary response that answers the question based on what is discussed in the texts.
        Do not mention anything outside of what is provided. Don't answer anything outside the context you are provided.
        If there isn't enough context, simply reply "This topic was not discussed previously"
        """
        self.SYSTEM_PROMPT = self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS

    def get_prompt(self, instruction):
        prompt_template =  self.B_INST + self.SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template

    def format_prompt(self, query, context):
        return '''
        ### Texts:
        {context}

        ### Question:
        {query}
        '''.format(context=context, query=query)

    def process_with_together_ai(self, person_data):
        prompt = self.format_prompt("Summarize the person's details.", person_data)
        result = self.client.completions.create(prompt=prompt, model="text-davinci-003")
        return result.choices[0].text.strip()
