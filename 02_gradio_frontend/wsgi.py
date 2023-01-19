from main import gradio_app

if __name__ == "__main__":
    gradio_app.launch(share=False,debug=True, server_name="0.0.0.0", server_port=7000)
