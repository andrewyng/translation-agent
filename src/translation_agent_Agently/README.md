# INSTRUCTION

This dir `translation_agent_Agently` is a replication of one chunk translation workflow using [Agently development framework](https://github.com/Maplemx/Agently) with some extra features:

- You can switch models easily by changing the settings in `./app.py` to test the workflow
- Clearly expression of workflow using Agently Workflow in `./workflow.py`
- You can edit prompt yaml files in `./prompt_yamls` to optimize prompt instructions without coding, these prompt yaml files will be loaded into model request when the workflow start
- Add a command line user input interface (enter 3 times to ensure programe know source text input is over)

# HOW TO RUN

```shell
git clone git@github.com:Maplemx/translation-agent.git
cd translation-agent/src/translation-agent-Agently
pip install -r requirements.txt
python app.py
```