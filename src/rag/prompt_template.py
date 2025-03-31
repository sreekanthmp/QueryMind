from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate
)


class PromptManager:
    @staticmethod
    def get_prompt_template():
        """Create and return the improved prompt template with history 
        awareness."""
        return ChatPromptTemplate(
            input_variables=["context", "question"],
            messages=[
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["context", "question"],
                        template=("""\
                            You are an AI assistant for answering user 
                            questions. 
                            Answer strictly based on the following retrieved 
                            context to generate a clear, structured response.
                            
                            ---
                            ## **Instructions**
                   
                            1. If the retrieved **context** contains relevant 
                                information:
                               - Use it to answer the question directly.
                               - If multiple versions exist, separate them 
                                  clearly.
                                - Don't summerize unless mentioned
                            
                            2.  If the context contains sufficient and 
                                  relevant information:
                                - Analyze the context to identify distinct 
                                  pieces of information or versions 
                                  of the answer related to the question.
                                - If multiple versions or scenarios are 
                                  present (e.g., 
                                    instructions for different tools or 
                                  methods), provide a separate response 
                                  for each version explicitly.
                                - Structure your response for clarity, with 
                                  clear headings and subheadings to 
                                  distinguish between different scenarios.
                            3. If the question is irrelevent return 
                                  "no response"
                            ---
                            ## **Response Format**
                           - Use markdown-style formatting for clarity:
                            - For example: `# Heading`, `## Subheading`, `- 
                                  Bullet points`.
                            - Provide a detailed, clear, and concise answer 
                                  for each version or scenario, organizing 
                                  the response appropriately.

                            ### Example Structure:
                            - **Version 1:** [Brief description]
                            - Url to the version or context if any available 
                                in the context
                            - Detailed explanation or steps for this scenario.

                            - **Version 2:** [Brief description]
                            - Url to the version or context if any available 
                                in the context
                            - Detailed explanation or steps for this scenario.

                            ---
                          
                            ## **Retrieved Context**
                            {context}

                            ---
                            ## **User Question**
                            {question}

                           
                            """)
                    )
                )
            ],
        )
