# AI Visibility Lab - Tracker

Asks fixed questions to AI models many times and records which brands and domains they name.

## Commands (run from this folder, with the venv active)
    python -m avl check                      # validate config, no API calls
    python -m avl run --limit-questions 2 --runs 1   # small test
    python -m avl run                        # full batch
    python -m avl run --resume RUN_ID        # continue / retry failed answers
    python -m pytest                         # run tests (pip install -r requirements-dev.txt)

## Files
- config/settings.toml   run settings and model names (verify model names and limits yourself)
- config/questions.toml  questions and brands (mark real target brands with target = true)
- data/raw/answers.csv   one row per answer, full text kept
- data/raw/runs/*.json   one manifest per run (settings, counts, status)
- logs/tracker.log       log file

## Honest limits
API models are not the ChatGPT app or Google AI Overviews. Domains are "as stated by the model".
Answers vary between runs, so report counts, not single results.
