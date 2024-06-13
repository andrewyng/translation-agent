import Agently

translation_workflow = Agently.Workflow()

@translation_workflow.chunk()
def user_input(inputs, storage):
  # get agent_factory from start data and put it into storage
  storage.set("agent_factory", inputs["default"]["agent_factory"])
  # get translation settings from user input
  source_language = input("[Source Language]: ")
  target_language = input("[Target Language]: ")
  country = input("[Country](Optional): ")
  source_text = []
  # get multi lines text from user input
  print("[Source Text]:")
  enter_count = 0
  while True:
    line = input()
    if line:
      source_text.append(line)
      enter_count = 0
    elif enter_count >= 2:
      break
    else:
      enter_count += 1
  source_text = "\n".join(source_text)
  # pass all user input data to next chunk
  return {
    "source_language": source_language,
    "target_language": target_language,
    "country": country,
    "source_text": source_text,
  }

@translation_workflow.chunk()
def initial_translation(inputs, storage):
  # get data from user_input chunk
  source_language = inputs["default"]["source_language"]
  target_language = inputs["default"]["target_language"]
  country = inputs["default"]["country"]
  source_text = inputs["default"]["source_text"]

  # create translation agent
  agent_factory = storage.get("agent_factory")
  translation_agent = agent_factory.create_agent()
  # set role (system-prompt-like)
  translation_agent.set_role(f"You are an expert linguist, specializing in translation from {source_language} to {target_language}.")
  # load commands from P-YAML to agent and get initial translation result
  print("[Initial Translation]:")
  initial_translation = (
    translation_agent
      .load_yaml_prompt(
        path = "./prompt_yamls/initial_translation.yaml",
        variables = {
          "source_language": source_language,
          "target_language": target_language,
          "source_text": source_text,
        }
      )
      # Streaming Output
      .on_delta(lambda data: print(data, end=""))
      .start()
  )

  # save translation_agent to storage because it will be reused in improve_translation chunk
  storage.set("translation_agent", translation_agent)

  # save other data to storage because they will be reused in next chunks
  storage.set("source_language", source_language)
  storage.set("target_language", target_language)
  storage.set("country", country)
  storage.set("initial_translation", initial_translation)
  return

@translation_workflow.chunk()
def reflect_on_translation(inputs, storage):
  # get data from storage
  source_language = storage.get("source_language")
  target_language = storage.get("target_language")
  country = storage.get("country")
  source_text = storage.get("source_text")
  initial_translation = storage.get("initial_translation")

  # create another agent because it has different role settings
  agent_factory = storage.get("agent_factory")
  reflect_on_translation_agent = agent_factory.create_agent()
  # set role (system-prompt-like)
  reflect_on_translation_agent.set_role(f"You are an expert linguist specializing in translation from {source_language} to {target_language}. You will be provided with a source text and its translation and your goal is to improve the translation.")
  # load commands from P-YAML to agent
  (
    reflect_on_translation_agent
      .load_yaml_prompt(
        path = "./prompt_yamls/reflect_on_translation.yaml",
        variables = {
          "source_language": source_language,
          "target_language": target_language,
          "source_text": source_text,
          "initial_translation": initial_translation
        }
      )
  )
  # append additional instruction when `country` is not empty
  if country and len(country) > 0:
    reflect_on_translation_agent.instruct([f"\nThe final style and tone of the translation should match the style of {target_language} colloquially spoken in {country}."])
  # give commands of current task and get reflection result
  print("\n[Reflection]:")
  reflection = (
    reflect_on_translation_agent
      # Streaming Output
      .on_delta(lambda data: print(data, end=""))
      .start()
  )
  # save reflection result to storage
  storage.set("reflection", reflection)
  return

@translation_workflow.chunk()
def improve_translation(inputs, storage):
  # get data from storage
  source_language = storage.get("source_language")
  target_language = storage.get("target_language")
  source_text = storage.get("source_text")
  initial_translation = storage.get("initial_translation")
  reflection = storage.get("reflection")

  # reuse translation agent
  translation_agent = storage.get("translation_agent")
  # get final translation result
  print("\n[Final Translation]:")
  final_translation = (
    translation_agent
      .load_yaml_prompt(
        path = "./prompt_yamls/improve_translation.yaml",
        variables = {
          "source_language": source_language,
          "target_language": target_language,
          "source_text": source_text,
          "initial_translation": initial_translation,
          "reflection": reflection,
        }
      )
      # Streaming Output
      .on_delta(lambda data: print(data, end=""))
      .start()
  )
  return final_translation

(
  translation_workflow.chunks["start"]
    .connect_to(translation_workflow.chunks["user_input"])
    .connect_to(translation_workflow.chunks["initial_translation"])
    .connect_to(translation_workflow.chunks["reflect_on_translation"])
    .connect_to(translation_workflow.chunks["improve_translation"])
    .connect_to(translation_workflow.chunks["end"])
)