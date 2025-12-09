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
        logger.info("IntentAgent: Analyzing brief for Playbook requirements...")
        system_prompt = """You are an expert AI Intent Analyst for Conversational Agents. 
        Your goal is to analyze the user's brief to extract the core components needed for a 'Playbooks' style agent.

        Analyze the brief and extract:
        1. Agent Role: What is the high-level identity? (e.g., "Helpful Cymbal Assistant")
        2. Persona Traits: Specific behavioral rules, tone (e.g., "Empathetic", "Direct"), and prohibited topics.
        3. Core Requirements: Non-negotiable rules (e.g., "Must verify identity", "Never share internal IDs").
           - CRITICAL: If the brief implies code generation, data processing, or technical output:
             a) If NO specific programming language is provided, you MUST explicitly add "All code output must be in Scala Spark" to the Core Requirements.
             b) You MUST explicitly add a requirement: "The final output MUST be a complete, executable program/script in a markdown code block. Do not just describe the code."
        4. User Journeys (Taskflows): What are the main tasks? (e.g., "Check Bill", "Transfer Funds").
           - If it is a coding task, the Taskflow MUST include a step named "Generate Program" or similar.
        5. Tools Needed: inferred tools based on tasks (e.g., `get_account_details`, `transfer_agent`).

        Output a structured analysis focusing on these elements."""
        return self.generate(system_prompt, brief)

class DesignAgent(Agent):
    def design(self, analysis, output_format="xml"):
        logger.info(f"DesignAgent: Designing structure for {output_format}...")
        
        if output_format == "markup":
            system_prompt = """You are an expert PromptML Architect. 
            Based on the intent analysis, design the structure for a PromptML (.pml) file.
            
            The structure MUST follow strict PromptML syntax:
            
            @prompt
                @context (The Agent's Role/Identity) @end
                
                @objective (The Goal or main functionality) @end
                
                @instructions
                    @step (Specific instructions/rules) @end
                @end
                
                @examples
                    @example
                        @input ... @end
                        @output ... @end
                    @end
                @end
                
                @constraints
                    @tone ... @end
                    (Include "Scala Spark" and "Code Block" rules here if applicable)
                @end
                
                @metadata ... @end
            @end
            
            Define the content for each section based on the analysis.
            Output a blueprint for the .pml file.
            """
        else:
            # Default to XML Playbooks
            system_prompt = """You are an expert Playbook Architect. 
            Based on the intent analysis, design the hierarchical XML structure for the Playbook Prompt.
            
            The structure MUST follow this hierarchy:
            
            <role>
            (High level identity)
            </role>
            
            <persona>
            (Behavioral rules, tone, prohibited topics)
            </persona>
            
            <requirements>
            (Mandatory security/identity/scope rules)
            (Include the Code Generation and Scala Spark rules if applicable)
            </requirements>
            
            <global_directives>
            (Interrupt handlers like "Speak to Human", "Language Change")
            </global_directives>
            
            <taskflow>
            <subtask name="...">
                <step name="...">
                    <action>...</action>
                </step>
                <handler name="...">...</handler>
            </subtask>
            </taskflow>
            
            <examples>
            (One Happy Path, One Edge Case/Unhappy Path)
            (If a coding task, the Happy Path example MUST show the Agent outputting a code block)
            </examples>

            Define the specific content for each section based on the analysis.
            Output a detailed blueprint."""
            
        return self.generate(system_prompt, analysis)

class CreatorAgent(Agent):
    def create_prompt(self, design, original_brief, output_format="xml"):
        logger.info(f"CreatorAgent: Generating final prompt for {output_format}...")
        
        if output_format == "markup":
            system_prompt = """You are a Master PromptML Writer.
            Your task is to generate the final, production-ready PromptML code using the provided design and brief.
            
            STRICT SYNTAX RULES (PromptML):
            1. All sections start with @section_name and end with @end.
            2. Main sections: @prompt, @context, @objective, @instructions, @examples, @constraints, @metadata.
            3. Instruction steps: @step [instruction text] @end.
            4. Examples: @example -> @input ... @end -> @output ... @end.
            5. Constraints: @length, @tone, @difficulty.
            6. Comments use #.
            
            CRITICAL CONTENT RULES:
            - If the task is technical, ensure "Scala Spark" is enforced in @constraints or @instructions.
            - If code is required, ensure instructions explicitly ask for a Markdown Code Block.
            
            Generate the COMPLETE .pml code.
            """
        else:
            # Default to XML Playbooks
            system_prompt = """You are a Master Playbooks Prompt Writer.
            Your task is to generate the final, production-ready system prompt using the provided design and brief.
            
            STRICT RULES FROM THE GUIDE:
            1. Hierarchy: Use the XML tags: <role>, <persona>, <requirements>, <global_directives>, <taskflow>, <examples>.
            2. Naming: 
            - XML attributes: lowercase in quotes (e.g., name="subtask name").
            - Tools: snake_case (e.g., `get_account_balance`).
            - States: SCREAMING_SNAKE_CASE.
            3. Formatting:
            - Use **bold** for key concepts/variables (e.g., **'$customerName'**).
            - Use UPPERCASE for mandatory terms (MUST, ALWAYS, NEVER).
            4. Taskflow Logic:
            - Use 'Invoke tool...' for tool calls.
            - Use IF...THEN...ELSE logic for branching.
            - Use <handler> for subtask-specific edge cases.
            5. Examples:
            - clearly label 'Agent:' and 'User:'.
            - Show tool calls as: [Tool: tool_name: 'outcome'].

            Generate the COMPLETE prompt in Markdown/XML.
            """
            
        user_input = f"Original Brief: {original_brief}\n\nPrompt Design Blueprint:\n{design}"
        return self.generate(system_prompt, user_input)
