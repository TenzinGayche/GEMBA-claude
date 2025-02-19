import ipdb
import json
import re
from collections import defaultdict

def apply_template(template, data):
    if isinstance(template, str):
        return template.format(**data)
    elif isinstance(template, list):
        prompt = []
        for conversation_turn in template:
            p = conversation_turn.copy()
            p['content'] = p['content'].format(**data)
            prompt.append(p)
        return prompt
    else:
        raise ValueError(f"Unknown template type {type(template)}")

def parse_broken_json(x):
    improved_translation = ""
    errors = defaultdict(list)
    if '"errors": ' in x and "improved translation" in x:
        data = x.split('", "errors": ')
        if len(data) != 2:
            return {"improved translation": improved_translation, "errors": errors}
        # from data[0] parse improved translation
        improved_translation = data[0].split('"improved translation": "')[1]
        # remove last character from data[1]
        data[1] = data[1][:-1]

        try:
            errors = json.loads(data[1])
        except:
            # just try to get error count
            words = re.findall(r'\b\w+\b', data[1].lower())
            keywords = ['critical', 'major', 'minor']

            last_key = None
            for word in words:
                if word in keywords:
                    last_key = word
                elif last_key is not None and word == "class":
                    errors[last_key].append({"class": "other"})

    return {"improved translation": improved_translation, "errors": errors}


def parse_error_class(error):
    # parse error from error description, errors are ['accuracy', 'fluency', 'locale convention', 'style', 'terminology', 'non-translation', 'other']
    #  locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use),
    class_name = "unknown"
    if "accuracy" in error:
        class_name = "accuracy"
        for subclass in ["addition", "mistranslation", "omission", "untranslated text"]:
            if subclass in error:
                class_name = f"accuracy-{subclass}"
    elif "fluency" in error:
        class_name = "fluency"
        for subclass in ["character encoding", "grammar", "inconsistency", "punctuation", "register", "spelling"]:
            if subclass in error:
                class_name = f"fluency-{subclass}"
    elif "locale convention" in error:
        class_name = "locale convention"
        for subclass in ["currency", "date", "name", "telephone", "time"]:
            if subclass in error:
                class_name = f"locale convention-{subclass}"
    elif "style" in error:
        class_name = "style"
    elif "terminology" in error:
        class_name = "terminology"
        for subclass in ["inappropriate", "inconsistent"]:
            if subclass in error:
                class_name = f"terminology-{subclass}"
    elif "non-translation" in error:
        class_name = "non-translation"
    elif "other" in error:
        class_name = "other"

    return class_name


def parse_mqm_answer(x, list_mqm_errors=False, full_desc=True):
    if x is None:
        return None

    x = str(x)
    if x.startswith('{"improved translation"'):
        try:
            x = json.loads(x)
        except:
            x = parse_broken_json(x)
        errors = x["errors"]
    else:
        x = x.lower()
        errors = {'critical': [], 'major': [], 'minor': []}
        error_level = None
        section_errors = {
            'critical': False,
            'major': False,
            'minor': False
        }
        
        # Split by any number of consecutive newlines
        import re
        lines = re.split(r'\n+', x)
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            if line.endswith(':'):
                current_section = line[:-1].lower()
                if current_section in ['critical', 'major', 'minor']:
                    error_level = current_section
                continue
            
            if error_level is not None:
                if "no-error" in line or "no error" in line:
                    continue
                
                if any(phrase in line for phrase in [
                    "the translation appears to be",
                    "it correctly conveys",
                    "the grammar and terminology",
                    "there are no obvious errors"
                ]):
                    continue
                    
                if "non-translation" in line:
                    errors["critical"].append(line)
                    section_errors['critical'] = True
                elif line:
                    errors[error_level].append(line)
                    section_errors[error_level] = True

    error_classes = defaultdict(list)
    final_score = 0
    error_counter = 0
    
    for error_level in ['critical', 'major', 'minor']:
        if error_level not in errors:
            continue
            
        for error in errors[error_level]:
            if error_counter < 5 and not list_mqm_errors and section_errors[error_level]:
                if error_level == 'critical':
                    final_score += 25
                elif error_level == 'major':
                    final_score += 5
                else:
                    final_score += 1
                error_counter += 1

            if full_desc:
                error_classes[error_level].append(error)
            else:
                class_name = parse_error_class(error)
                error_classes[error_level].append(class_name)
                    
    if final_score > 25:
        final_score = 25

    if list_mqm_errors:
        return error_classes
    else:
        return -final_score

def mqm_fewshot(few_shots):
    prompts = [
    ]

    template = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."""
   
    for shot in few_shots:
        prompts.append({
            "role": "user",
            "content": template.format(**shot)
        })
        answer = shot['answer']

        prompts.append({
            "role": "assistant",
            "content": answer
        })

    prompts.append({
            "role": "user",
            "content": template
        })

    return prompts


few_shots = {
    "ende": {
            "source_lang": "English",
            "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
            "target_lang": "German",
            "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
            "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
        },
    "encs": {
            "source_lang": "English",
            "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
            "target_lang": "Czech",
            "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
            "answer": """Critical:
no-error
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""",
        },
    "zhen": {
            "source_lang": "Chinese",
            "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
            "target_lang": "English",
            "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
            "answer": """Critical:
accuracy/addition - "of high-speed rail"
Major:
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""",
        },
}

TEMPLATE_GEMBA_MQM = mqm_fewshot([few_shots['ende'], few_shots['encs'], few_shots['zhen']])

