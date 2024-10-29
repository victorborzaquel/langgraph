from os import environ, getcwd, path

from dotenv import load_dotenv


def run_preload():
    if "HTTP_PROXY" in environ:
        del environ["HTTP_PROXY"]
    if "HTTPS_PROXY" in environ:
        del environ["HTTPS_PROXY"]

    environ["MONGO_PROXY"] = "None"

    custom_temp_dir = path.join(getcwd(), "tmp")

    environ["TMPDIR"] = custom_temp_dir
    environ["TEMP"] = custom_temp_dir

    load_dotenv()
    print("----Preload done----")


run_preload()
