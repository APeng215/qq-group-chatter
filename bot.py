import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

from qq_group_chatter.app import create_default_orchestrator
from qq_group_chatter.plugins.chat import setup_orchestrator

nonebot.init()
driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)
nonebot.load_plugin("qq_group_chatter.plugins.chat")

orchestrator = create_default_orchestrator()
setup_orchestrator(orchestrator)


@driver.on_startup
async def start_services() -> None:
    long_term_memory = getattr(orchestrator, "_long_term_memory")
    if hasattr(long_term_memory, "start"):
        await long_term_memory.start()


@driver.on_shutdown
async def stop_services() -> None:
    long_term_memory = getattr(orchestrator, "_long_term_memory")
    if hasattr(long_term_memory, "stop"):
        await long_term_memory.stop()


if __name__ == "__main__":
    nonebot.run()
