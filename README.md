<div align="center">
  
<a href="https://github.com/ImPavloh/rvc-tts" target="_blank"><img src="https://img.shields.io/github/license/impavloh/rvc-tts?style=for-the-badge&logo=github&logoColor=white"></a>
<a href="https://twitter.com/ImPavloh" target="_blank"><img src="https://img.shields.io/badge/Pavloh-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white"></a>

<h1>🎙️ An AI-Powered Text-to-Speech Bot 🤖💬</h1>

<h3><a href="https://github.com/ImPavloh/rvc-tts-discord-bot">Try VoiceMe! the Discord Bot</a></h3>
<a href="https://github.com/ImPavloh/rvc-tts-discord-bot"><img alt="English" src="https://i.imgur.com/hc6AbYN.png" width="50px" style="border-top-left-radius: 25px; border-bottom-left-radius: 25px;"></a>


</div>

## 🌐 Online use

> Go to Hugging Face 🤗
 
- **https://huggingface.co/spaces/ImPavloh/RVC-TTS-Demo**

    *Yep, that's all!*

## 🤖 Discord use

> Go to this repository and follow the README steps.

- **https://github.com/ImPavloh/rvc-tts-discord-bot**

## 🛠️ Local use

1. Clone the repository 🗂️
```bash
git clone https://github.com/ImPavloh/rvc-tts
```

2. Change to the project directory 📁
```bash
cd rvc-tts
```

3. Install the necessary dependencies 📦
```bash
pip install -r requirements.txt
```

4. Download the Hubert base model from **[here](https://huggingface.co/spaces/ImPavloh/RVC-TTS-Demo/resolve/main/hubert_base.pt)**. Save the downloaded file to the root folder of the project.

5. Add your RVC models following the next format 📂
```Swift
└── Models
    └── ModelName
        └── ModelName
            ├── File.pth
            └── File.index
```

6. Configure the **config.ini** file with your preferences ⚙️
   
7. Run the main script 🚀
```bash
python app.py
```

## 📄 Important Files

🗂️  `models/`: Folder that should contain the voice models that will be used for text-to-speech conversion. If everything is correct, the bot will automatically detect the models and information files for the program will be generated.

📑  `requirements.txt`: File containing all the Python dependencies needed for the bot to function.

🤖  `app.py`: Python main script. This will start the gradio web-server.

## ⚡ Optimizations

Everything is optimized to ensure minimal RAM and CPU usage. The audio conversion uses the "PM" method, which is the fastest and only requires a CPU, without the need for a GPU. This makes running the bot on virtually any device/server possible.

## 📝 License

By using this project, you agree to the [license](https://github.com/ImPavloh/rvc-tts/blob/main/LICENSE).
