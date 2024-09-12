# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:55:00 2024

@author: ayhan
"""

import chainlit as cl


@cl.step(type="tool")
async def tool():
    # Fake tool
    await cl.sleep(2)
    return "Response from the tool!"


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    final_answer = await cl.Message(content="").send()

    # Call the tool
    final_answer.content = await tool()

    await final_answer.update()