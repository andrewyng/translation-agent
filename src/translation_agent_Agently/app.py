import Agently
from workflow import translation_workflow

# Choose model you like and set settings according:
# http://agently.tech/features/model_request.html
agent_factory = (
  Agently.AgentFactory()
    .set_settings("current_model", "OAIClient")
    .set_settings("model.OAIClient.url", "https://api.deepseek.com/")
    .set_settings("model.OAIClient.auth", { "api_key": "" })
    .set_settings("model.OAIClient.options", { "model": "deepseek-chat" })
)

translation_workflow.start({
  "agent_factory": agent_factory,
})