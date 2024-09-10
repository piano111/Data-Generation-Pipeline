# 数据生成Pipeline

数据生成pipeline是根据instructpix2pix的人类指令指导图像编辑数据集生成过程构建的，主要包含两个部分：第一部分是人工编写的文本数据{图像编辑前的描述，编辑指令，图像编辑后的描述}，使用这些文本数据对大语言模型进行微调，使得大语言模型可以生成大量的格式化文本指令数据；第二部分是根据生成的文本指令数据进行配对图像数据的生成；最终得到的数据集可以组成为{编辑前的图像，编辑指令，编辑后的图像}

## 文本指令生成

文本指令首先需要几百条文本数据（项目中目前使用的是100条)，这些文本数据可以人工编撰也可以使用大语言模型进行生成，然后进行数据清洗；

我们使用大语言模型进行初始的文本数据生成，进行数据清洗后将{图像编辑前的描述，编辑指令，图像编辑后的描述}这种数据分别放入3个文本文档，使用[GetTextData.py](https://github.com/piano111/Data-Generation-Pipeline/blob/main/InstructGetData/GetTextData.py)脚本转化为json文件，然后通过[GetNewJson.py](https://github.com/piano111/Data-Generation-Pipeline/blob/main/InstructGetData/GetNewJson.py)脚本将json文件转化为用来微调大语言模型的jsonl文件；

微调训练过程：运行[训练脚本](https://github.com/piano111/Data-Generation-Pipeline/blob/main/InstructGetData/train.py)即可获得微调后的模型；微调推理过程：运行[推理脚本](https://github.com/piano111/Data-Generation-Pipeline/blob/main/InstructGetData/inference.py)获得文本指令和编辑后的描述，并保存为json文件



## 配对图像数据生成

配对图像数据生成主要使用prompt to prompt技术，这个技术可以有效解决diffusion模型生成不稳定的问题，并提供了编辑方法；我们使用refinement方法根据文本数据集生成编辑前的图像和编辑后的图像；运行[prompt to prompt.py](https://github.com/piano111/Data-Generation-Pipeline/blob/main/prompt-to-prompt/prompt2prompt.py)文件即可生成相应的配对的编辑前后的图像
