#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import os

record_path = '/home/hieung1707/'
record_file = 'sample.mp3'
API_KEY = "a93d3a127f9c4ee89015f0f46ca02d47"
API_TOKEN_BOT = "720052620754ec80808f6f86510358fb"


def speech_recognition(audio):
    url = "https://api.openfpt.vn/fsr"
    response = requests.post(url,
                             data=audio,
                             headers={'api_key': API_KEY,
                                      'Content-Type': ''})
    r = response.json()
    status = r['status']
    text = ''
    # print(json.dumps(r))
    if status == 0:
        text = r['hypotheses'][0]['utterance']
    return status, text


def audio_to_byte():
    file_path = record_path + record_file
    audio_in_byte = ''
    max_size = 1024
    with open(file_path, 'rb') as f:
        while True:
            buf = f.read(max_size)
            if not buf:
                break
            audio_in_byte += buf

    return audio_in_byte


def get_intent(text):
    url = "https://v3-api.fpt.ai/api/v3/predict/intent"
    d = {}
    d['content'] = text
    # d['save_history'] = 'false'
    json_data = json.dumps(d)
    # print(json_data)

    response = requests.post(url,
                             headers={'Authorization': API_TOKEN_BOT},
                             data=json_data)
    r = response.json()
    # print(json.dumps(r))
    status = r['status']['code']
    intent = ''
    if status == 200:
        intent = r['data']['intents'][0]['label']
    return status, intent


def stt():
    status, text = speech_recognition(audio_to_byte())
    text_received = 'status: ' + str(status) + '; text:' + text
    if text == '':
        return 'không có text', ""
    status, intent = get_intent(text)
    intent_received = 'status: ' + str(status) + '; intent: ' + intent
    return text_received, intent_received

stt()