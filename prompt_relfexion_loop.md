
04-28

接下来我要新增加一个可回溯的模式，具体行为是基于不可回溯模式（noback）一直向前，直到hit tool.maxstep。

# agent_r1/tool/tool_env_retro_noback.py:896-899
if current_step >= self.maxstep:
    current_message += StepFail.format(n=self.maxstep)  # "Pathway fails because the maximum number of reaction steps 30 has been reached..."
    current_message += FailBack                          # 提示模型回溯
    self.back_flag = True

引导模型回溯，接着基于回溯的state继续进行逆合成预测

可回溯模式可参考 @Agent-r1-temp/agent_r1/src/agent_ray_trainer_retro.py，但请注意，我们需要构建的框架与 @Agent-r1-temp/agent_r1/src/agent_ray_trainer_retro.py 中的回溯模式有很大区别：（1）agent_ray_trainer_retro.py每一步都可选择回溯，这会导致该方法一直在浅层打转，无法生成长的逆合成路径，从而导致最终效果收到很大影响。
（2）我们想要构建的可回溯模式在到达tool.maxstep之前，一直使用的都是noback模式，只有在到达tool.maxstep时，才会出现回溯的选项，这样会强迫其生成长的逆合成路径，避免在浅层打转，从而极大的提高最终的performance