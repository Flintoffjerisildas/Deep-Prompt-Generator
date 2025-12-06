import os
import argparse
from dotenv import load_dotenv
from agents import IntentAgent, DesignAgent, CreatorAgent
from utils import setup_logging, save_to_md

# Load environment variables
load_dotenv()

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Deep Prompt Generator System")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate a Deep Prompt from a brief.")
    parser.add_argument("--brief", type=str, help="The user brief for the task.")
    parser.add_argument("--file", type=str, help="Path to a file containing the brief.")
    args = parser.parse_args()

    brief = ""
    if args.brief:
        brief = args.brief
    elif args.file:
        try:
            with open(args.file, "r") as f:
                brief = f.read()
        except Exception as e:
            logger.error(f"Error reading brief file: {e}")
            return
    else:
        logger.error("No brief provided. Use --brief or --file.")
        print("Please provide a brief using --brief 'text' or --file 'path/to/file.txt'")
        return

    logger.info(f"Received brief: {brief[:50]}...")

    try:
        # Initialize Agents
        api_key_1 = os.environ.get("agent-1-api-key")
        api_key_2 = os.environ.get("agent-2-api-key")
        api_key_3 = os.environ.get("agent-3-api-key")

        if not all([api_key_1, api_key_2, api_key_3]):
            logger.error("Missing one or more API keys in .env")
            print("Error: Please ensure agent-1-api-key, agent-2-api-key, and agent-3-api-key are set in .env")
            return

        intent_agent = IntentAgent(api_key=api_key_1)
        design_agent = DesignAgent(api_key=api_key_2)
        creator_agent = CreatorAgent(api_key=api_key_3)

        # Step 1: Intent Analysis
        logger.info("Step 1: Analyzing Intent...")
        intent_analysis = intent_agent.analyze(brief)
        logger.info("Intent Analysis complete.")

        # Step 2: Prompt Design
        logger.info("Step 2: Designing Prompt...")
        prompt_design = design_agent.design(intent_analysis)
        logger.info("Prompt Design complete.")

        # Step 3: Prompt Creation
        logger.info("Step 3: Creating Final Prompt...")
        final_prompt = creator_agent.create_prompt(prompt_design, brief)
        logger.info("Final Prompt created.")

        # Save Output
        save_to_md(final_prompt, "deep_prompt_output.md")
        print("\n--- Deep Prompt Generation Complete ---\n")
        print(f"Output saved to: {os.path.abspath('deep_prompt_output.md')}")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
