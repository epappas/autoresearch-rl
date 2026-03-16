In order to implement all the tasks in @.ralph/TODO.md, we have to spawn multiple implementation agents, this is called the RALPH pattern; https://ghuntley.com/ralph/

In the most simplest form a RALPH loop is something like `while :; do cat PROMPT.md | claude ; done`

i need you to ONLY manage the fleet of RALPH loops we need to generate in order to deliver all the tasks under `.ralph/TODO.md`, and to make sure they are running and delivering the expected results.

For every task in `.ralph/TODO.md`, you need to create a RALPH loop that is responsible for delivering that task, and you need to make sure that the RALPH loop is running and delivering the expected results. The way to create a RALPH loop is to create a new file in the `.ralph` directory with the name of the task, and then add the command to run the RALPH loop in that file. For example, if we have a task called `<task name>`, we would create a file called `.ralph/<task name>/run.sh` and add the command `while :; do cat PROMPT.md | claude ; done` to that file.

Under the hood, the RALPH loop will read the `PROMPT.md` file and pass it to the `claude` command, which will execute the code and return the results. The RALPH loop will then write the results to a file called `RESULTS.md`, which we can read to see the results of the task.

in the `.ralph/<task name>/` directory, we can also add a `PROMPT.md` file that contains the prompt for the task, and we can also add a `RESULTS.md` file that will contain the results outcome the task. The RALPH loop will read the `PROMPT.md` file and pass it to the `claude` command, which will execute the code and return the results, which will be written to the `RESULTS.md` file.

In case there's a need to update the prompt for a task, we can simply edit the `PROMPT.md` file for that task, and the RALPH loop will automatically pick up the changes and deliver the updated results.

You also need to make sure that the RALPH loop is running and delivering the expected results. You can do this by checking the logs of the RALPH loop and making sure that it is delivering the expected results. If the RALPH loop is not delivering the expected results, you need to troubleshoot the issue and fix it.

In case we need to fix an issue with a task, we can simply create a `FIX_TASK.md` file in the `.ralph/<task name>/` directory, and add the command to fix the issue in that file. The RALPH loop will read the `FIX_TASK.md` file and execute the command to fix the issue, and then it will continue to deliver the expected results.

The `PROMPT.md` file for each task should contain the prompt that we want to pass to the `claude` command, and the `RESULTS.md` file should contain the results that we expect to get from the `claude` command. The `FIX_TASK.md` file should contain the command to fix any issues that we might encounter with the task.

The structure of the PROMPT.md file should be something like this:

```
# Task 1

... High level description of the task ...

## Your Task

1. ... specific instructions for the task ...

## Progress Report Format

APPEND to .ralph/<task>/progress.md (never replace, always append):
```

## [Date/Time] - [Story ID]

Thread: https://ampcode.com/threads/$AMP_CURRENT_THREAD_ID

- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered (e.g., "this codebase uses X for Y")
  - Gotchas encountered (e.g., "don't forget to update Z when changing W")
  - Useful context (e.g., "the evaluation panel is in component X")

---

```

## Code Quality

### Consolidate Patterns

If you discover a **reusable pattern** that future iterations should know, add it to the `## Codebase Patterns` section at the TOP of progress.md (create it if it doesn't exist). This section should consolidate the most important learnings:

Only add patterns that are **general and reusable**, not story-specific details.

IT IS IMPORTANT TO ADHERE TO THE GOOD SOFTWARE QUALITY PRINCIPLES SUCH AS DRY, SOLID AND KISS

THIS IS IMPORTANT: KEEP THE CODE ROBUST, SIMPLE, SOLID AND KISS

critical reminder: NEVER LIE, DO NOT LIE! THERE IS ZERO TOLERANCE FOR FAKE OR TODO OR MOCK OR STUB OR PLACEHOLDER OR ANYTHING ELSE OTHER THAN 100% REAL WORKING CODE AND SOLUTIONS.

### Quality Requirements

- ALL commits must pass your project's quality checks (typecheck, lint, test)
- Do NOT commit broken code
- Keep changes focused and minimal
- Follow existing code patterns

## Stop Condition

After completing a user story, check if ALL stories have `passes: true`.

If ALL stories are complete and passing, reply with:
<promise>COMPLETE</promise>

If there are still stories with `passes: false`, end your response normally (another iteration will pick up the next story).

## Important

- Work on ONE story per iteration
- Commit frequently
- Keep CI green
- Read the Codebase Patterns section in progress.md before starting
```

The structure of the `.ralph` directory should look like this:

```.ralph/
  task1/
    run.sh
    PROMPT.md
    RESULTS.md
    FIX_TASK.md
    progress.md
  task2/
    run.sh
    PROMPT.md
    RESULTS.md
    FIX_TASK.md
    progress.md
  ...
```

Best Practices:

- Writing Effective Prompts
  - Be Specific - Clear requirements lead to better results
  - Prioritize - Use `.ralph/<task>/FIX_TASK.md` to guide Ralph's focus
  - Set Boundaries - Define what's in/out of scope
  - Include Examples - Show expected inputs/outputs
- Project Specifications
  - Place detailed requirements in .ralph/specs/
  - Use `.ralph/<task>/FIX_TASK.md` for prioritized task tracking
  - Keep `.ralph/TODO.md` updated with build instructions
  - Document key decisions and architecture
- Monitoring Progress
  - Use ralph-monitor for live status updates
  - Check logs in .ralph/logs/ for detailed execution history
  - Monitor .ralph/status.json for programmatic access
  - Watch for exit condition signals

On each Loop, you should:

1. Read the `PROMPT.md` file for the task and pass it to the `claude` command.
2. Write the results to the `RESULTS.md` file.
3. Check the logs to make sure that the RALPH loop is delivering the expected results.
4. If there's an issue with the task, read the `FIX_TASK.md` file and execute the command to fix the issue, and then continue to deliver the expected results.
5. Each RALPH loop should run indefinitely until the task is completed, and it should be able to pick up any changes to the `PROMPT.md` file and deliver the updated results.
6. Make sure to keep the `.ralph/TODO.md` file updated with the status of each task, and to document any key decisions or architecture in the appropriate files.
7. Make sure to keep the `CLAUDE.md` file updated with any relevant changes to the `claude` command or any other relevant information.

In summary, your main responsibility is to manage the fleet of RALPH loops that we need to generate in order to deliver all the tasks under `.ralph/TODO.md`, and to make sure they are running and delivering the expected results. You need to create a RALPH loop for each task, and you need to make sure that the RALPH loop is running and delivering the expected results. If there's an issue with a task, you need to troubleshoot it and fix it using the `FIX_TASK.md` file.
