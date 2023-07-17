<div align="center">
  
![Logo](https://i.imgur.com/Dh8vuLR.png)
  
<a href="https://github.com/ImPavloh/rvc-tts" target="_blank"><img src="https://img.shields.io/github/license/impavloh/rvc-tts?style=for-the-badge&logo=github&logoColor=white"></a>
<a href="https://twitter.com/ImPavloh" target="_blank"><img src="https://img.shields.io/badge/Pavloh-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white"></a>

<h1>🎙️ An AI-Powered Text-to-Speech 🤖💬</h1>
</div>

## 🛠️ Local installation

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

4. Add your RVC models following the next format 📂
```Swift
└── Models
    └── ModelName
        └── ModelName
            ├── File.pth
            └── File.index
```

5. Run the main script 🚀
```bash
python app.py
```

## 📄 Important Files

🗂️  `models/`: Folder that should contain the voice models that will be used for text-to-speech conversion. If everything is correct, the bot will automatically detect the models and information files for the program will be generated.

📑  `requirements.txt`: File containing all the Python dependencies needed for the bot to function.

🤖  `app.py`: Python script. This will start the bot with the configuration and models.

## ⚡ Optimizations

Everything is optimized to ensure minimal RAM and CPU usage. The audio conversion uses the "PM" method, which is the fastest and only requires a CPU, without the need for a GPU. This makes running the bot on virtually any device/server possible.

## 📝 License

By using this project, you agree to the [license](https://github.com/ImPavloh/rvc-tts/blob/main/LICENSE).
