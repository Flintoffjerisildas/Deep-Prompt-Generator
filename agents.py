import os
from groq import Groq
import logging

logger = logging.getLogger("DeepPromptSystem")

class Agent:
    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, system_prompt, user_prompt):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in agent generation: {e}")
            raise

class IntentAgent(Agent):
    def analyze(self, brief):
        logger.info("IntentAgent: Analyzing brief...")
        system_prompt = """You are an expert AI Intent Analyst. Your goal is to deeply understand the user's request for a Scala coding task.
        Break down the user's brief into:
        1. Core Objective: What is the main goal?
        2. Key Requirements: What are the specific constraints and functional requirements?
        3. Implicit Needs: What is likely needed but not explicitly stated (e.g., error handling, performance)?
        4. Scala Specifics: What Scala features or libraries are relevant?
        
        Output a structured analysis."""
        return self.generate(system_prompt, brief)

class DesignAgent(Agent):
    def design(self, analysis):
        logger.info("DesignAgent: Designing prompt structure...")
        system_prompt = """You are an expert Prompt Engineer and Architect. Based on the provided intent analysis, design the structure of the perfect 'Deep Prompt' to give to an LLM.
        The goal is to create a prompt that will 'one-shot' the task.
        
        Design the prompt sections:
        1. Persona/Role: Who should the LLM act as?
        2. Context: What background info is needed?
        3. Task Description: How should the task be phrased for maximum clarity?
        4. Constraints & Guidelines: What rules must be followed (especially for Scala)?
        5. Output Format: How should the code be presented?
        6. Examples (Few-Shot): What kind of examples would help?
        
        Output a blueprint/outline for the final prompt."""
        return self.generate(system_prompt, analysis)

class CreatorAgent(Agent):
    def create_prompt(self, design, original_brief):
        logger.info("CreatorAgent: Creating final prompt...")
        system_prompt = """You are a Master Prompt Writer. Your task is to synthesize the provided prompt design and the original brief into a single, high-quality, 'Deep Prompt'.
        
        The final output must be a ready-to-use prompt.
        It should be detailed, clear, and optimized for high-performance LLMs.
        Focus on Scala best practices (functional programming, type safety, etc.).
        
        Structure the prompt using Markdown headers.
        Ensure the tone is professional and authoritative.
        Include a section for 'Chain of Thought' or 'Step-by-Step' reasoning if beneficial.
        """
        user_input = f"Original Brief: {original_brief}\n\nPrompt Design:\n{design}"
        return self.generate(system_prompt, user_input)
