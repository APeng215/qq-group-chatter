import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

from qq_group_chatter.app import create_default_application
from qq_group_chatter.logging_config import configure_runtime_logging
from qq_group_chatter.memory_dashboard import setup_memory_dashboard

nonebot.init()
configure_runtime_logging()
driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)
nonebot.load_plugin("qq_group_chatter.plugins.chat")

from qq_group_chatter.plugins.chat import setup_orchestrator

application = create_default_application()
setup_orchestrator(application.orchestrator)
setup_memory_dashboard(driver, application)


@driver.on_startup
async def start_services() -> None:
    await application.start()


@driver.on_shutdown
async def stop_services() -> None:
    await application.stop()


if __name__ == "__main__":
    nonebot.run()
