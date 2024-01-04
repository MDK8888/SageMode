from typing import Union

class IOTypes:
    LanguageModeling = {"text":str, "parameters":Union[dict, None]}
    ImageModeling = {"base64":str, "parameters":Union[dict, None]}
    Logits = {"logits": list[float], "parameters":Union[dict, None]}
