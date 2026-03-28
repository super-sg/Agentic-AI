"""
Autonomous Research Agent — Main Entry Point.

An AI-powered research agent built with LangChain that:
1. Takes a topic as input
2. Searches the web and Wikipedia for relevant information
3. Analyzes and synthesizes findings
4. Generates a structured research report

Usage:
    python research_agent.py "Impact of AI in Healthcare"
    python research_agent.py  # Interactive mode
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from tools import get_all_tools
from report_generator import generate_report

# ─── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.3
MAX_TOOL_ITERATIONS = 12

# ─── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research analyst. Your task is to conduct thorough research
on the given topic using the available tools and produce a comprehensive, well-structured report.

## Research Instructions

For the given topic, you MUST:
1. **Search the web** using the web_search tool to find current information, recent developments, statistics, and expert opinions.
2. **Search Wikipedia** using the wikipedia tool to gather foundational knowledge, definitions, and historical context.
3. **Make at least 4-5 different searches** across both tools to gather comprehensive information.
4. **Synthesize all gathered information** into a coherent, well-structured report.

IMPORTANT: Use the tools first to gather information BEFORE writing the report.
Make multiple searches with different queries to get comprehensive coverage.

## Final Report Format

After gathering enough information, write your FINAL report in Markdown format with these exact sections:

## Table of Contents

1. [Introduction](#introduction)
2. [Key Findings](#key-findings)
3. [Challenges](#challenges)
4. [Future Scope](#future-scope)
5. [Conclusion](#conclusion)

---

## Introduction
(Provide a comprehensive introduction to the topic. Include background context,
why this topic is important, and what the report covers. 2-3 paragraphs.)

## Key Findings
(Present 5-7 major findings from your research. Each finding should be a subsection
with a bold heading and 2-3 sentences of explanation. Include statistics and data
where available.)

## Challenges
(Discuss 3-5 major challenges or concerns related to the topic. Be specific and
provide examples.)

## Future Scope
(Explore 3-5 future possibilities, emerging trends, and potential developments.
Include timelines or predictions where applicable.)

## Conclusion
(Summarize the key takeaways in 2-3 paragraphs. Provide a balanced perspective.)

## References
(List the key sources you consulted during research.)
"""


def create_research_agent():
    """
    Create and return the LLM with tool-calling capabilities.
    """
    # Validate API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        print("   Please set it in a .env file or export it:")
        print("   export GOOGLE_API_KEY='your_key_here'")
        sys.exit(1)

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=api_key,
        max_output_tokens=8192,
    )

    # Get tools and bind to LLM
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, tools


def _invoke_with_retry(llm, messages, max_retries=5):
    """
    Invoke the LLM with automatic retry on rate limit (429) errors.
    Uses exponential backoff with delays extracted from error messages.
    """
    import re

    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Extract retry delay from error message
                retry_match = re.search(r"retry in (\d+\.?\d*)", error_str, re.IGNORECASE)
                wait_time = float(retry_match.group(1)) + 5 if retry_match else (30 * (attempt + 1))
                print(f"  ⏳ Rate limited. Waiting {wait_time:.0f}s before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise non-rate-limit errors

    raise Exception("Max retries exceeded due to rate limiting. Please try again later.")


def run_agent_loop(llm_with_tools, tools, topic: str) -> tuple[str, list]:
    """
    Run the agent in a tool-calling loop until it produces a final response.

    Returns:
        Tuple of (final_output_text, list_of_tool_calls_made)
    """
    # Build tool lookup
    tool_map = {t.name: t for t in tools}
    tool_calls_log = []

    # Initialize conversation
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research the following topic and generate a comprehensive report:\n\n{topic}"),
    ]

    for iteration in range(MAX_TOOL_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{MAX_TOOL_ITERATIONS} ---")

        # Get LLM response with retry logic for rate limits
        response = _invoke_with_retry(llm_with_tools, messages)
        messages.append(response)

        # Check if there are tool calls
        if not response.tool_calls:
            # No more tool calls — the model has produced its final answer
            print("✅ Agent finished research. Generating final report...")
            return response.content, tool_calls_log

        # Execute each tool call
        from langchain_core.messages import ToolMessage

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"  🔧 Using tool: {tool_name}")
            print(f"     Query: {tool_args}")

            try:
                # Execute the tool
                tool_result = tool_map[tool_name].invoke(tool_args)
                # Truncate very long results to avoid context overflow
                if isinstance(tool_result, str) and len(tool_result) > 4000:
                    tool_result = tool_result[:4000] + "\n\n[... truncated for brevity ...]"
                print(f"     ✓ Got results ({len(str(tool_result))} chars)")
            except Exception as e:
                tool_result = f"Error using {tool_name}: {str(e)}"
                print(f"     ✗ Error: {e}")

            tool_calls_log.append({
                "tool": tool_name,
                "query": tool_args,
                "result_length": len(str(tool_result)),
            })

            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_id,
            ))

        # Small delay to respect rate limits
        time.sleep(2)

    # If we hit max iterations, ask for final summary
    print("⚠️  Max iterations reached. Requesting final report...")
    messages.append(HumanMessage(
        content="You have gathered enough information. Please write the final report NOW with all the sections specified."
    ))
    final_response = _invoke_with_retry(llm_with_tools, messages)
    return final_response.content, tool_calls_log


def research_topic(topic: str) -> str:
    """
    Run the research agent on a given topic and generate a report.

    Args:
        topic: The research topic.

    Returns:
        Path to the generated report file.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Autonomous Research Agent")
    print(f"{'='*60}")
    print(f"📝 Topic: {topic}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🔧 Tools: Web Search (DuckDuckGo) + Wikipedia")
    print(f"{'='*60}\n")
    print("🔍 Starting research... (this may take 1-3 minutes)\n")

    # Create agent
    llm_with_tools, tools = create_research_agent()

    # Run the agent loop
    output, tool_calls = run_agent_loop(llm_with_tools, tools, topic)

    # Show tool usage summary
    if tool_calls:
        print(f"\n{'='*60}")
        print(f"📊 Research Summary")
        print(f"{'='*60}")
        print(f"   Total tool calls: {len(tool_calls)}")
        tool_counts = {}
        for call in tool_calls:
            tool_counts[call["tool"]] = tool_counts.get(call["tool"], 0) + 1
        for tool_name, count in tool_counts.items():
            print(f"   • {tool_name}: {count} calls")
        print()

    # Generate the report
    print("📝 Generating report...")
    report_path = generate_report(topic, output)
    print(f"✅ Report saved to: {report_path}")
    print(f"{'='*60}\n")

    return report_path


def main():
    """Main entry point — accepts topic from CLI args or interactive input."""
    if len(sys.argv) > 1:
        # Topic from command-line argument
        topic = " ".join(sys.argv[1:])
    else:
        # Interactive input
        print("\n🔬 Autonomous Research Agent (LangChain)")
        print("=" * 45)
        topic = input("\nEnter a research topic: ").strip()
        if not topic:
            print("❌ No topic provided. Exiting.")
            sys.exit(1)

    report_path = research_topic(topic)
    return report_path


if __name__ == "__main__":
    main()
