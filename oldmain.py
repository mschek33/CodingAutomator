import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please create a config.json file with the following structure:")
        print(json.dumps({
            "openrouter_api_key": "your-openrouter-api-key-here",
            "model": "openai/gpt-4o-mini",
            "starting_project_directory": "C:/Projects"
        }, indent=4))
        sys.exit(1)

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

    required_fields = ["openrouter_api_key", "model", "starting_project_directory"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        print(f"Error: Missing required configuration fields: {', '.join(missing_fields)}")
        sys.exit(1)

    return config


def validate_config(config: dict) -> bool:
    """Validate configuration values."""
    valid = True

    if config["openrouter_api_key"] == "your-openrouter-api-key-here":
        print("Error: OpenRouter API key has not been set. Please update config.json")
        valid = False

    project_dir = Path(config["starting_project_directory"])
    if not project_dir.exists():
        print(f"Error: Starting project directory does not exist: {project_dir}")
        valid = False
    elif not project_dir.is_dir():
        print(f"Error: Starting project path is not a directory: {project_dir}")
        valid = False

    return valid


def natural_sort_key(path: Path) -> tuple:
    """Generate a sort key for natural sorting of filenames like 1.1, 1.2, 1.10."""
    filename = path.stem  # Get filename without extension
    parts = re.split(r'(\d+)', filename)
    # Convert numeric parts to integers for proper sorting
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def find_stories(project_dir: Path) -> list[Path]:
    """Find all story files in the docs/stories folder, sorted naturally."""
    stories_dir = project_dir / "docs" / "stories"

    if not stories_dir.exists():
        print(f"Warning: Stories directory not found: {stories_dir}")
        return []

    story_files = sorted(stories_dir.glob("*.md"), key=natural_sort_key)
    return story_files


def find_prd_files(project_dir: Path) -> list[Path]:
    """Find all markdown files in the docs/prd folder, sorted naturally."""
    prd_dir = project_dir / "docs" / "prd"

    if not prd_dir.exists():
        print(f"Warning: PRD directory not found: {prd_dir}")
        return []

    prd_files = sorted(prd_dir.glob("*.md"), key=natural_sort_key)
    return prd_files


def detect_epic_list_file(client: OpenAI, model: str, prd_files: list[Path]) -> Path | None:
    """Use AI to detect which PRD file contains the epic LIST (not details)."""
    if not prd_files:
        return None

    # Build a list of filenames for the AI to analyze
    file_list = "\n".join([f"- {f.name}" for f in prd_files])

    prompt = f"""Analyze these PRD folder filenames and identify the ONE file that contains the epic LIST.
This is the file that lists all epics/features at a high level (NOT the detailed epic descriptions).
It might be named something like "epic-list.md", "epics.md", "5-epic-list.md", etc.

Files in docs/prd:
{file_list}

Return ONLY the single filename that contains the epic list. If none found, return "NONE".
Do not include any other text or explanation."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You analyze project documentation structure. Respond only with a single filename, no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )

        response_text = response.choices[0].message.content.strip()

        if response_text == "NONE" or not response_text:
            return None

        # Find the matching file
        filename = response_text.strip().lstrip("- ")
        for prd_file in prd_files:
            if prd_file.name == filename:
                return prd_file

        return None

    except Exception as e:
        print(f"Error detecting epic list file: {e}")
        return None


def extract_epics_from_list(client: OpenAI, model: str, epic_list_file: Path) -> list[dict]:
    """Use AI to extract individual epics from the epic list file."""
    try:
        with open(epic_list_file, "r", encoding="utf-8") as f:
            content = f.read()

        prompt = f"""Analyze this epic list file and extract each epic.
For each epic, provide the epic number/ID and title.

File content:
{content[:8000]}

Return a JSON array of objects with "number" and "title" fields. Example:
[{{"number": "1", "title": "User Authentication"}}, {{"number": "2", "title": "Dashboard"}}]

Return ONLY the JSON array, no other text."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract structured data from documents. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2000
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            epics = json.loads(json_match.group())
            return epics

        return []

    except Exception as e:
        print(f"Error extracting epics: {e}")
        return []


def run_codex_review_for_story(project_dir: Path, story_path: str) -> bool:
    """Run Codex CLI to review implementation of a story."""

    print(f"\nRunning Codex review for: {story_path}")
    print()

    original_dir = os.getcwd()
    try:
        # Change to project directory
        os.chdir(str(project_dir))
        print(f"Working directory: {os.getcwd()}")

        review_prompt = f"Review the implementation of the story at {story_path}. Read the story file to understand all tasks and acceptance criteria. Check that all tasks marked [x] are actually implemented in the codebase. Verify the code follows the patterns in docs/architecture. Check that unit tests exist and cover the implementation. Look for any bugs, missing error handling, or incomplete implementations. If you find issues, fix any bugs or incomplete implementations and add missing tests. IMPORTANT: Apply all changes directly to the files - do not just propose changes or ask for confirmation. Make the edits now. Report what you found and what you fixed."

        # Use subprocess.list2cmdline for Windows compatibility
        cmd = ['codex', 'exec', '--skip-git-repo-check', '--dangerously-bypass-approvals-and-sandbox', '--json', review_prompt]

        print(f"Running: codex exec ...")

        # On Windows, use shell=True to find codex.cmd
        use_shell = sys.platform == 'win32'
        if use_shell:
            # Build command string for Windows shell
            cmd_str = subprocess.list2cmdline(cmd)
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

        # Stream output line by line
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event_type = data.get("type", "")
                # Handle various Codex event types
                if event_type == "item.completed":
                    item = data.get("item", {})
                    item_type = item.get("type", "")
                    if item_type == "agent_message":
                        print(item.get("text", ""), flush=True)
                    elif item_type == "reasoning":
                        print(f"[Thinking] {item.get('text', '')}", flush=True)
                elif event_type == "thread.started":
                    print(f"[Started thread: {data.get('thread_id', 'unknown')}]", flush=True)
                elif event_type == "turn.started":
                    print("[Processing...]", flush=True)
                elif event_type == "turn.completed":
                    usage = data.get("usage", {})
                    print(f"[Turn complete - tokens: {usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out]", flush=True)
            except json.JSONDecodeError:
                # Print non-JSON output directly
                print(line, flush=True)

        process.wait()
        os.chdir(original_dir)

        print(f"\nCodex exit code: {process.returncode}")
        return process.returncode == 0

    except FileNotFoundError as e:
        print(f"Error: Could not find codex executable: {e}")
        print("Make sure codex-cli is installed: npm install -g @openai/codex")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"Error running Codex: {type(e).__name__}: {e}")
        os.chdir(original_dir)
        return False


def run_codex_review_for_epic(project_dir: Path, epic_number: str, epic_title: str) -> bool:
    """Run Codex CLI to review stories created for an epic."""

    print(f"\nRunning Codex review for Epic {epic_number}: {epic_title}")
    print()

    original_dir = os.getcwd()
    try:
        # Change to project directory
        os.chdir(str(project_dir))
        print(f"Working directory: {os.getcwd()}")

        review_prompt = f"Review the stories created for epic {epic_number} ({epic_title}) in docs/stories. Check if any tasks or requirements from the epic definition in docs/prd have been missed. Compare against docs/architecture, docs/front-end-spec, and docs/prd for completeness. For each story file for epic {epic_number}: verify all acceptance criteria from the epic are covered, check that unit and integration testing tasks are included, identify any missing tasks or subtasks. IMPORTANT: Apply all changes directly to the story files - do not just propose changes or ask for confirmation. Make the edits now using file write operations. Report what was missing and what you added."

        # Use subprocess.list2cmdline for Windows compatibility
        cmd = ['codex', 'exec', '--skip-git-repo-check', '--dangerously-bypass-approvals-and-sandbox', '--json', review_prompt]

        print(f"Running: codex exec ...")

        # On Windows, use shell=True to find codex.cmd
        use_shell = sys.platform == 'win32'
        if use_shell:
            # Build command string for Windows shell
            cmd_str = subprocess.list2cmdline(cmd)
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

        # Stream output line by line
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event_type = data.get("type", "")
                # Handle various Codex event types
                if event_type == "item.completed":
                    item = data.get("item", {})
                    item_type = item.get("type", "")
                    if item_type == "agent_message":
                        print(item.get("text", ""), flush=True)
                    elif item_type == "reasoning":
                        print(f"[Thinking] {item.get('text', '')}", flush=True)
                elif event_type == "thread.started":
                    print(f"[Started thread: {data.get('thread_id', 'unknown')}]", flush=True)
                elif event_type == "turn.started":
                    print("[Processing...]", flush=True)
                elif event_type == "turn.completed":
                    usage = data.get("usage", {})
                    print(f"[Turn complete - tokens: {usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out]", flush=True)
            except json.JSONDecodeError:
                # Print non-JSON output directly
                print(line, flush=True)

        process.wait()
        os.chdir(original_dir)

        print(f"\nCodex exit code: {process.returncode}")
        return process.returncode == 0

    except FileNotFoundError as e:
        print(f"Error: Could not find codex executable: {e}")
        print("Make sure codex-cli is installed: npm install -g @openai/codex")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"Error running Codex: {type(e).__name__}: {e}")
        os.chdir(original_dir)
        return False


def run_claude_code_for_epic(project_dir: Path, epic_number: str, epic_title: str) -> bool:
    """Run Claude Code as a subprocess to create stories from an epic using SM role."""

    print(f"Running Claude Code for Epic {epic_number}: {epic_title}")
    print(f"Working directory: {project_dir}")
    print()

    try:
        # Change to project directory
        original_dir = os.getcwd()
        os.chdir(str(project_dir))

        # Step 1: Load the SM (Scrum Master) role and get session ID
        print("Step 1: Loading Scrum Master role...")
        init_prompt = "Read .bmad-core/agents/sm.md and adopt that scrum master role."
        cmd1 = f'claude -p "{init_prompt}" --allowedTools "Read" --output-format json'

        result1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

        if result1.returncode != 0:
            print(f"Error loading SM role: {result1.stderr}")
            os.chdir(original_dir)
            return False

        # Extract session ID from JSON response
        try:
            response = json.loads(result1.stdout)
            session_id = response.get("session_id")
            print(f"Session ID: {session_id}")
        except json.JSONDecodeError:
            print("Could not parse session response")
            os.chdir(original_dir)
            return False

        # Step 2: Continue session with the create stories command
        print(f"\nStep 2: Creating stories for Epic {epic_number}...")
        create_prompt = f"Create the stories for epic {epic_number} ({epic_title}), using the info in docs/architecture and docs/front-end-spec and docs/prd as appropriate. Use sub agents where you can. Make sure to include unit and integration testing in the stories."
        cmd2 = f'claude -p "{create_prompt}" --resume "{session_id}" --allowedTools "Bash,Read,Edit,Write,Glob,Grep" --output-format stream-json --verbose'

        process = subprocess.Popen(
            cmd2,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )

        # Stream output line by line
        for line in process.stdout:
            try:
                data = json.loads(line)
                if data.get("type") == "assistant":
                    content = data.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            print(block.get("text", ""), end="", flush=True)
                elif data.get("type") == "result":
                    print(f"\n\nResult: {data.get('result', 'Done')}")
            except json.JSONDecodeError:
                print(line, end="", flush=True)

        process.wait()

        # Change back
        os.chdir(original_dir)

        return process.returncode == 0

    except Exception as e:
        print(f"Error running Claude Code: {e}")
        return False


def parse_story(story_path: Path) -> dict:
    """Parse a story file and extract key information."""
    with open(story_path, "r", encoding="utf-8") as f:
        content = f.read()

    story_data = {
        "filename": story_path.name,
        "filepath": str(story_path),
        "content": content,
        "status": extract_status(content),
        "tasks": extract_tasks(content),
        "title": extract_title(content, story_path.name)
    }

    return story_data


def extract_status(content: str) -> str:
    """Extract the status from story content."""
    status_patterns = [
        r"##\s*Status\s*\n+\s*(Draft|Approved|InProgress|Review|Done)",
        r"\*\*Status\*\*:\s*(Draft|Approved|InProgress|Review|Done)",
        r"Status:\s*(Draft|Approved|InProgress|Review|Done)",
    ]

    for pattern in status_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

    return "Unknown"


def extract_title(content: str, filename: str) -> str:
    """Extract the title from story content or filename."""
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()

    return filename.replace(".md", "").replace("-", " ").replace("_", " ")


def extract_tasks(content: str) -> dict:
    """Extract tasks and their completion status."""
    completed_pattern = r"- \[x\]"
    incomplete_pattern = r"- \[ \]"

    completed = len(re.findall(completed_pattern, content, re.IGNORECASE))
    incomplete = len(re.findall(incomplete_pattern, content))
    total = completed + incomplete

    return {
        "total": total,
        "completed": completed,
        "incomplete": incomplete,
        "completion_percentage": round((completed / total * 100) if total > 0 else 0, 1)
    }


def analyze_story_with_openai(client: OpenAI, model: str, story_data: dict, max_retries: int = 3) -> dict:
    """Use OpenAI to analyze the story's progress and status."""
    prompt = f"""Analyze this BMAD story and provide a progress assessment.

Story: {story_data['title']}
Current Status: {story_data['status']}
Tasks: {story_data['tasks']['completed']}/{story_data['tasks']['total']} completed ({story_data['tasks']['completion_percentage']}%)

Story Content:
{story_data['content'][:4000]}

Based on the story content, tasks completion, and status, provide:
1. Overall completion status (Complete/Incomplete/Not Started)
2. A brief summary (1-2 sentences) of what remains to be done
3. Any blockers or concerns identified

IMPORTANT: Do not take into account deferred tasks when assessing completion. Tasks marked as deferred or moved to future stories should be ignored.

Respond in this exact JSON format:
{{
    "completion_status": "Complete|Incomplete|Not Started",
    "summary": "brief summary here",
    "blockers": "any blockers or empty string if none",
    "confidence": "High|Medium|Low"
}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a project analyst reviewing BMAD methodology stories. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=500
            )

            response_content = response.choices[0].message.content

            if not response_content:
                if attempt < max_retries - 1:
                    print(f"    Retry {attempt + 1}/{max_retries}: Empty response, retrying...")
                    time.sleep(1)
                    continue
                else:
                    print(f"    ERROR: Empty response from API after {max_retries} attempts")
                    return {
                        "completion_status": "Error",
                        "summary": "Empty response from API",
                        "blockers": "",
                        "confidence": "N/A"
                    }

            response_text = response_content.strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = json.loads(response_text)

            return analysis

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries}: JSON parse error, retrying...")
                time.sleep(1)
                continue
            print(f"    ERROR: Failed to parse JSON: {str(e)}")
            print(f"    Response was: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
            return {
                "completion_status": "Error",
                "summary": f"Failed to parse response",
                "blockers": "",
                "confidence": "N/A"
            }
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            return {
                "completion_status": "Error",
                "summary": f"Failed to analyze: {str(e)}",
                "blockers": "",
                "confidence": "N/A"
            }

    # Fallback if loop completes without returning
    return {
        "completion_status": "Error",
        "summary": "Failed after all retries",
        "blockers": "",
        "confidence": "N/A"
    }


def load_story_creation_report(output_path: Path) -> dict:
    """Load existing story creation report and return a dict of epic_number -> status."""
    existing = {}

    if not output_path.exists():
        return existing

    try:
        tree = ET.parse(output_path)
        root = tree.getroot()

        for epic_elem in root.findall(".//Epic"):
            epic_number = epic_elem.get("number")
            if epic_number:
                existing[epic_number] = {
                    "number": epic_number,
                    "title": epic_elem.findtext("Title", ""),
                    "stories_created": epic_elem.findtext("StoriesCreated", "false") == "true",
                    "creation_reviewed": epic_elem.findtext("CreationReviewed", "false") == "true",
                    "created_at": epic_elem.findtext("CreatedAt", ""),
                    "reviewed_at": epic_elem.findtext("ReviewedAt", "")
                }

        print(f"Loaded {len(existing)} existing epic records from creation report")
    except Exception as e:
        print(f"Warning: Could not load story creation report: {e}")

    return existing


def save_story_creation_report(epics_status: list, output_path: Path) -> None:
    """Save story creation report to XML."""
    root = ET.Element("StoryCreationReport")
    root.set("generated", datetime.now().isoformat())
    root.set("total_epics", str(len(epics_status)))

    # Summary
    summary = ET.SubElement(root, "Summary")
    created_count = sum(1 for e in epics_status if e.get("stories_created"))
    reviewed_count = sum(1 for e in epics_status if e.get("creation_reviewed"))
    ET.SubElement(summary, "StoriesCreated").text = str(created_count)
    ET.SubElement(summary, "CreationReviewed").text = str(reviewed_count)

    # Epics
    epics_elem = ET.SubElement(root, "Epics")
    for epic in epics_status:
        epic_elem = ET.SubElement(epics_elem, "Epic")
        epic_elem.set("number", str(epic["number"]))

        ET.SubElement(epic_elem, "Title").text = epic.get("title", "")
        ET.SubElement(epic_elem, "StoriesCreated").text = str(epic.get("stories_created", False)).lower()
        ET.SubElement(epic_elem, "CreationReviewed").text = str(epic.get("creation_reviewed", False)).lower()
        ET.SubElement(epic_elem, "CreatedAt").text = epic.get("created_at", "")
        ET.SubElement(epic_elem, "ReviewedAt").text = epic.get("reviewed_at", "")

    # Pretty print
    xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"Story creation report saved to: {output_path}")


def load_existing_report(output_path: Path) -> dict:
    """Load existing XML report and return a dict of filename -> analysis."""
    existing = {}

    if not output_path.exists():
        return existing

    try:
        tree = ET.parse(output_path)
        root = tree.getroot()

        for story_elem in root.findall(".//Story"):
            filename = story_elem.get("filename")
            if filename:
                tasks_elem = story_elem.find("Tasks")
                existing[filename] = {
                    "filename": filename,
                    "title": story_elem.findtext("Title", ""),
                    "status": story_elem.findtext("Status", "Unknown"),
                    "tasks": {
                        "total": int(tasks_elem.get("total", 0)) if tasks_elem is not None else 0,
                        "completed": int(tasks_elem.get("completed", 0)) if tasks_elem is not None else 0,
                        "incomplete": 0,
                        "completion_percentage": float(tasks_elem.get("percentage", 0)) if tasks_elem is not None else 0
                    },
                    "analysis": {
                        "completion_status": story_elem.findtext("CompletionStatus", "Unknown"),
                        "summary": story_elem.findtext("Summary", ""),
                        "blockers": story_elem.findtext("Blockers", ""),
                        "confidence": story_elem.findtext("Confidence", "N/A")
                    },
                    "implemented": story_elem.findtext("Implemented", "false") == "true",
                    "implemented_at": story_elem.findtext("ImplementedAt", ""),
                    "dev_reviewed": story_elem.findtext("DevReviewed", "false") == "true",
                    "dev_reviewed_at": story_elem.findtext("DevReviewedAt", "")
                }

        print(f"Loaded {len(existing)} existing results from previous report")
    except Exception as e:
        print(f"Warning: Could not load existing report: {e}")

    return existing


def generate_xml_report(stories_analysis: list, output_path: Path) -> None:
    """Generate an XML report of all stories' status."""
    root = ET.Element("StoriesReport")
    root.set("generated", datetime.now().isoformat())
    root.set("total_stories", str(len(stories_analysis)))

    # Summary section
    summary = ET.SubElement(root, "Summary")
    complete_count = sum(1 for s in stories_analysis if s["analysis"]["completion_status"] == "Complete")
    incomplete_count = sum(1 for s in stories_analysis if s["analysis"]["completion_status"] == "Incomplete")
    not_started_count = sum(1 for s in stories_analysis if s["analysis"]["completion_status"] == "Not Started")
    error_count = sum(1 for s in stories_analysis if s["analysis"]["completion_status"] == "Error")
    implemented_count = sum(1 for s in stories_analysis if s.get("implemented", False))
    reviewed_count = sum(1 for s in stories_analysis if s.get("dev_reviewed", False))

    ET.SubElement(summary, "Complete").text = str(complete_count)
    ET.SubElement(summary, "Incomplete").text = str(incomplete_count)
    ET.SubElement(summary, "NotStarted").text = str(not_started_count)
    ET.SubElement(summary, "Errors").text = str(error_count)
    ET.SubElement(summary, "Implemented").text = str(implemented_count)
    ET.SubElement(summary, "DevReviewed").text = str(reviewed_count)

    # Stories section
    stories_elem = ET.SubElement(root, "Stories")

    for story in stories_analysis:
        story_elem = ET.SubElement(stories_elem, "Story")
        story_elem.set("filename", story["filename"])

        ET.SubElement(story_elem, "Title").text = story["title"]
        ET.SubElement(story_elem, "Status").text = story["status"]
        ET.SubElement(story_elem, "CompletionStatus").text = story["analysis"]["completion_status"]

        tasks_elem = ET.SubElement(story_elem, "Tasks")
        tasks_elem.set("total", str(story["tasks"]["total"]))
        tasks_elem.set("completed", str(story["tasks"]["completed"]))
        tasks_elem.set("percentage", str(story["tasks"]["completion_percentage"]))

        ET.SubElement(story_elem, "Summary").text = story["analysis"]["summary"]
        ET.SubElement(story_elem, "Blockers").text = story["analysis"]["blockers"]
        ET.SubElement(story_elem, "Confidence").text = story["analysis"]["confidence"]

        # Implementation tracking
        ET.SubElement(story_elem, "Implemented").text = str(story.get("implemented", False)).lower()
        ET.SubElement(story_elem, "ImplementedAt").text = story.get("implemented_at", "")
        ET.SubElement(story_elem, "DevReviewed").text = str(story.get("dev_reviewed", False)).lower()
        ET.SubElement(story_elem, "DevReviewedAt").text = story.get("dev_reviewed_at", "")

    # Pretty print XML
    xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"\nXML report saved to: {output_path}")


def _run_story_creation(project_dir: Path, client: OpenAI, model: str, epics: list, creation_report_path: Path, creation_report: dict) -> None:
    """Run the story creation flow for epics."""
    # Check which epics already have stories
    stories_dir = project_dir / "docs" / "stories"
    existing_story_files = list(stories_dir.glob("*.md")) if stories_dir.exists() else []

    def epic_has_stories(epic_num: str) -> bool:
        for story_file in existing_story_files:
            if story_file.name.startswith(f"{epic_num}."):
                return True
        return False

    # Build status for all epics
    epics_status = []
    epics_to_create = []
    epics_to_review = []

    for epic in epics:
        epic_num = str(epic['number'])
        existing_record = creation_report.get(epic_num, {})

        status = {
            "number": epic_num,
            "title": epic['title'],
            "stories_created": existing_record.get("stories_created", False) or epic_has_stories(epic_num),
            "creation_reviewed": existing_record.get("creation_reviewed", False),
            "created_at": existing_record.get("created_at", ""),
            "reviewed_at": existing_record.get("reviewed_at", "")
        }

        if not status["stories_created"]:
            epics_to_create.append(epic)
        elif not status["creation_reviewed"]:
            epics_to_review.append(epic)

        epics_status.append(status)

    # Show status
    print()
    for status in epics_status:
        if status["stories_created"] and status["creation_reviewed"]:
            print(f"Epic {status['number']} ({status['title']}) - complete (created & reviewed)")
        elif status["stories_created"]:
            print(f"Epic {status['number']} ({status['title']}) - stories created, needs review")
        else:
            print(f"Epic {status['number']} ({status['title']}) - needs stories created")

    if not epics_to_create and not epics_to_review:
        print("\nAll epics have stories created and reviewed.")
        return

    print(f"\n{len(epics_to_create)} epic(s) need stories created")
    print(f"{len(epics_to_review)} epic(s) need review only")
    print()

    # Process each epic to create stories
    for i, epic in enumerate(epics_to_create, 1):
        print()
        print("=" * 60)
        print(f"CREATING STORIES FOR EPIC {i}/{len(epics_to_create)}: Epic {epic['number']} - {epic['title']}")
        print("=" * 60)

        success = run_claude_code_for_epic(project_dir, epic['number'], epic['title'])

        if success:
            print(f"Completed story creation for Epic {epic['number']}")
            # Update status
            for status in epics_status:
                if status["number"] == str(epic['number']):
                    status["stories_created"] = True
                    status["created_at"] = datetime.now().isoformat()
                    epics_to_review.append(epic)
                    break
            # Save progress
            save_story_creation_report(epics_status, creation_report_path)
        else:
            print(f"Failed to create stories for Epic {epic['number']}")
            cont = input("Continue with next epic? (y/n): ").strip().lower()
            if cont != 'y':
                print("Stopping story creation.")
                break

    # Run Codex review for all epics that need review
    if epics_to_review:
        print()
        print("=" * 60)
        print("CODEX REVIEW PHASE - Reviewing all created stories")
        print("=" * 60)

        for epic in epics_to_review:
            print()
            print(f"Reviewing Epic {epic['number']}: {epic['title']}")
            print("-" * 40)
            review_success = run_codex_review_for_epic(project_dir, epic['number'], epic['title'])
            if review_success:
                print(f"Codex review completed for Epic {epic['number']}")
                # Update status
                for status in epics_status:
                    if status["number"] == str(epic['number']):
                        status["creation_reviewed"] = True
                        status["reviewed_at"] = datetime.now().isoformat()
                        break
                # Save progress
                save_story_creation_report(epics_status, creation_report_path)
            else:
                print(f"Codex review had issues for Epic {epic['number']}")

    print()
    print("Story creation phase complete.")


def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("BMAD Story Progress Analyzer")
    print("=" * 60)
    print()

    # Load and validate configuration
    config = load_config()

    if not validate_config(config):
        print("\nPlease fix the configuration errors above.")
        sys.exit(1)

    project_dir = Path(config["starting_project_directory"])
    model = config["model"]

    print(f"Project Directory: {project_dir}")
    print(f"Model: {model}")
    print()

    # Initialize OpenRouter client
    client = OpenAI(
        api_key=config["openrouter_api_key"],
        base_url="https://openrouter.ai/api/v1"
    )

    # First, check if there are epics that still need stories created
    print("Checking for epics that need stories...")
    prd_files = find_prd_files(project_dir)

    if prd_files:
        epic_list_file = detect_epic_list_file(client, model, prd_files)

        if epic_list_file:
            epics = extract_epics_from_list(client, model, epic_list_file)

            if epics:
                # Load story creation report
                creation_report_path = project_dir / "story_creation_report.xml"
                creation_report = load_story_creation_report(creation_report_path)

                # Check which epics already have stories
                stories_dir = project_dir / "docs" / "stories"
                existing_story_files = list(stories_dir.glob("*.md")) if stories_dir.exists() else []

                def epic_has_stories(epic_num: str) -> bool:
                    for story_file in existing_story_files:
                        if story_file.name.startswith(f"{epic_num}."):
                            return True
                    return False

                # Find epics that need stories or review
                epics_needing_work = []
                for epic in epics:
                    epic_num = str(epic['number'])
                    existing_record = creation_report.get(epic_num, {})
                    has_stories = existing_record.get("stories_created", False) or epic_has_stories(epic_num)
                    is_reviewed = existing_record.get("creation_reviewed", False)

                    if not has_stories or not is_reviewed:
                        epics_needing_work.append({
                            "epic": epic,
                            "has_stories": has_stories,
                            "is_reviewed": is_reviewed
                        })

                if epics_needing_work:
                    print(f"Found {len(epics_needing_work)} epic(s) needing story creation or review")
                    for item in epics_needing_work:
                        epic = item["epic"]
                        if not item["has_stories"]:
                            print(f"  Epic {epic['number']} ({epic['title']}) - needs stories created")
                        else:
                            print(f"  Epic {epic['number']} ({epic['title']}) - needs review")

                    response = input("\nDo you want to continue with story creation? (y/n): ").strip().lower()
                    if response == 'y':
                        # Jump to story creation flow
                        _run_story_creation(project_dir, client, model, epics, creation_report_path, creation_report)
                        sys.exit(0)
                    print("Skipping story creation, proceeding to dev implementation...")
                    print()

    # Find stories for dev implementation
    print("Searching for stories...")
    story_files = find_stories(project_dir)

    if not story_files:
        print("No story files found in docs/stories folder and no epics to process.")
        sys.exit(0)

    print(f"Found {len(story_files)} story file(s)")
    print()

    # Load existing report if it exists
    output_path = project_dir / "stories_status_report.xml"
    existing_results = load_existing_report(output_path)

    # Process each story
    stories_analysis = []
    skipped_count = 0

    for i, story_path in enumerate(story_files, 1):
        filename = story_path.name

        # Skip if already analyzed
        if filename in existing_results:
            print(f"[{i}/{len(story_files)}] Skipping (cached): {filename}")
            stories_analysis.append(existing_results[filename])
            skipped_count += 1
            continue

        print(f"[{i}/{len(story_files)}] Analyzing: {filename}")

        story_data = parse_story(story_path)
        print(f"    Status: {story_data['status']}")
        print(f"    Tasks: {story_data['tasks']['completed']}/{story_data['tasks']['total']} ({story_data['tasks']['completion_percentage']}%)")

        # Analyze with OpenRouter
        analysis = analyze_story_with_openai(client, model, story_data)
        print(f"    AI Assessment: {analysis['completion_status']} (Confidence: {analysis['confidence']})")

        stories_analysis.append({
            "filename": story_data["filename"],
            "title": story_data["title"],
            "status": story_data["status"],
            "tasks": story_data["tasks"],
            "analysis": analysis
        })

        print()

    if skipped_count > 0:
        print(f"Skipped {skipped_count} already-analyzed stories")
        print()

    # Generate XML report
    generate_xml_report(stories_analysis, output_path)

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    complete = sum(1 for s in stories_analysis if s["analysis"]["completion_status"] == "Complete")
    incomplete = sum(1 for s in stories_analysis if s["analysis"]["completion_status"] == "Incomplete")
    not_started = sum(1 for s in stories_analysis if s["analysis"]["completion_status"] == "Not Started")

    print(f"Complete:    {complete}")
    print(f"Incomplete:  {incomplete}")
    print(f"Not Started: {not_started}")
    print(f"Total:       {len(stories_analysis)}")
    print()

    # Find incomplete stories to implement (not yet implemented or reviewed)
    incomplete_stories = [
        s for s in stories_analysis
        if s["analysis"]["completion_status"] in ["Incomplete", "Not Started"]
        and not s.get("implemented", False)
    ]

    # Find implemented but not reviewed stories
    needs_review_only = [
        s for s in stories_analysis
        if s.get("implemented", False) and not s.get("dev_reviewed", False)
    ]

    if not incomplete_stories and not needs_review_only:
        print("All stories are complete or implemented and reviewed!")
        return

    # Show status summary
    for s in stories_analysis:
        status_str = s["analysis"]["completion_status"]
        if s.get("dev_reviewed", False):
            status_str += " (implemented & reviewed)"
        elif s.get("implemented", False):
            status_str += " (implemented, needs review)"
        print(f"  {s['filename']}: {status_str}")

    print()
    print(f"Found {len(incomplete_stories)} stories to implement")
    print(f"Found {len(needs_review_only)} stories needing review only")
    print()

    # Ask user if they want to proceed with implementation
    response = input("Do you want to run Claude Code to implement these stories? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipping implementation phase.")
        return

    # Process each incomplete story
    implemented_stories = []
    for i, story in enumerate(incomplete_stories, 1):
        print()
        print("=" * 60)
        print(f"IMPLEMENTING STORY {i}/{len(incomplete_stories)}: {story['filename']}")
        print("=" * 60)

        story_path = f"docs/stories/{story['filename']}"
        success = run_claude_code_for_story(project_dir, story_path)

        if success:
            print(f"Completed implementation attempt for {story['filename']}")
            implemented_stories.append({"filename": story['filename'], "path": story_path})

            # Update story status in analysis
            for s in stories_analysis:
                if s["filename"] == story['filename']:
                    s["implemented"] = True
                    s["implemented_at"] = datetime.now().isoformat()
                    break

            # Save progress to XML
            generate_xml_report(stories_analysis, output_path)
        else:
            print(f"Failed to implement {story['filename']}")

            # Ask if user wants to continue
            cont = input("Continue with next story? (y/n): ").strip().lower()
            if cont != 'y':
                print("Stopping implementation.")
                break

    # Combine newly implemented stories with previously implemented but not reviewed
    all_stories_to_review = implemented_stories.copy()
    for s in needs_review_only:
        if not any(r["filename"] == s["filename"] for r in all_stories_to_review):
            all_stories_to_review.append({"filename": s["filename"], "path": f"docs/stories/{s['filename']}"})

    # Run Codex review for all stories that need review
    if all_stories_to_review:
        print()
        print("=" * 60)
        print("CODEX REVIEW PHASE - Reviewing all implemented stories")
        print("=" * 60)

        for story in all_stories_to_review:
            print()
            print(f"Reviewing: {story['filename']}")
            print("-" * 40)
            review_success = run_codex_review_for_story(project_dir, story['path'])
            if review_success:
                print(f"Codex review completed for {story['filename']}")

                # Update story status in analysis
                for s in stories_analysis:
                    if s["filename"] == story['filename']:
                        s["dev_reviewed"] = True
                        s["dev_reviewed_at"] = datetime.now().isoformat()
                        break

                # Save progress to XML
                generate_xml_report(stories_analysis, output_path)
            else:
                print(f"Codex review had issues for {story['filename']}")

    print()
    print("=" * 60)
    print("Implementation phase complete. Run the analyzer again to check status.")
    print("=" * 60)


def remove_story_from_cache(output_path: Path, filename: str) -> None:
    """Remove a story from the XML cache so it gets re-analyzed."""
    if not output_path.exists():
        return

    try:
        tree = ET.parse(output_path)
        root = tree.getroot()

        stories_elem = root.find("Stories")
        if stories_elem is not None:
            for story_elem in stories_elem.findall("Story"):
                if story_elem.get("filename") == filename:
                    stories_elem.remove(story_elem)
                    print(f"Removed {filename} from cache for re-analysis")
                    break

        # Save updated XML
        xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

    except Exception as e:
        print(f"Warning: Could not update cache: {e}")


def run_claude_code_for_story(project_dir: Path, story_path: str) -> bool:
    """Run Claude Code as a subprocess to implement a story using two-step process."""

    print(f"Running Claude Code for: {story_path}")
    print(f"Working directory: {project_dir}")
    print()

    try:
        # Change to project directory
        original_dir = os.getcwd()
        os.chdir(str(project_dir))

        # Step 1: Load the dev role and get session ID
        print("Step 1: Loading developer role...")
        init_prompt = "Read .bmad-core/agents/dev.md and adopt that developer role."
        cmd1 = f'claude -p "{init_prompt}" --allowedTools "Read" --output-format json'

        result1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)

        if result1.returncode != 0:
            print(f"Error loading dev role: {result1.stderr}")
            os.chdir(original_dir)
            return False

        # Extract session ID from JSON response
        try:
            response = json.loads(result1.stdout)
            session_id = response.get("session_id")
            print(f"Session ID: {session_id}")
        except json.JSONDecodeError:
            print("Could not parse session response")
            os.chdir(original_dir)
            return False

        # Step 2: Continue session with the develop-story command
        print(f"\nStep 2: Running *develop-story on {story_path}...")
        dev_prompt = f"*develop-story {story_path} - set the story from draft to in progress, use sub agents where you can, make sure you know whats in docs/architecture and docs/prd and docs/front-end-spec so you can reference where applicable. Please make sure that you mark each story task and sub task as complete when you complete it so that if you crash then we have proper context of what you have done. If you encounter any errors from previous implementations, fix them - do not ignore them as pre-existing issues."
        cmd2 = f'claude -p "{dev_prompt}" --resume "{session_id}" --allowedTools "Bash,Read,Edit,Write,Glob,Grep" --output-format stream-json --verbose'

        process = subprocess.Popen(
            cmd2,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )

        # Stream output line by line
        for line in process.stdout:
            try:
                data = json.loads(line)
                if data.get("type") == "assistant":
                    content = data.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            print(block.get("text", ""), end="", flush=True)
                elif data.get("type") == "result":
                    print(f"\n\nResult: {data.get('result', 'Done')}")
            except json.JSONDecodeError:
                print(line, end="", flush=True)

        process.wait()

        # Change back
        os.chdir(original_dir)

        return process.returncode == 0

    except Exception as e:
        print(f"Error running Claude Code: {e}")
        return False


if __name__ == "__main__":
    main()
