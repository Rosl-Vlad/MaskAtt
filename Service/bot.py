import yaml
import torch
import telebot

from os.path import join
from telebot import types
from hashlib import sha256

from src import Implementation

token = open("tg_token.txt", 'r')
bot = telebot.TeleBot(token.read())

d = {
    1: "✅",
    0: "❌",
}


def resolve_attrs(old_attrs, select_attr):
    if select_attr == 1 or select_attr == 2 or select_attr == 3:
        for i in range(1, 4):
            old_attrs[i] = 0
    return old_attrs


def image_clf_keyboard(message, attrs, attrs_name):
    keyboard = types.InlineKeyboardMarkup()
    for i, at in enumerate(attrs):
        key = types.InlineKeyboardButton(text=attrs_name[i] + " - " + d[int(attrs[i].item())],
                                         callback_data=attrs_name[i])
        keyboard.add(key)

    question = 'Select attribute'
    bot.send_message(message.from_user.id, text=question, reply_markup=keyboard)


if __name__ == "__main__":
    f = open("config.yaml", 'r')
    config = yaml.safe_load(f)
    f.close()

    i = Implementation(config)

    print("Model ready!")


    @bot.message_handler(content_types=['photo', "text"])
    def get_text_messages(message):
        global attr_a
        global image_name

        if message.text:
            try:
                attr_b = attr_a.clone()
                changes = message.text.split(", ")
                for c in changes:
                    att, value = c.split(":")
                    attr_b[0][i.d_attrs[att]] = float(value)

                render(attr_b)
                bot.send_photo(message.from_user.id,
                               photo=open(join(config["tg_bot"]["image_path_save"], "gen_" + image_name), 'rb'))
            except:
                bot.send_message(message.from_user.id, "Sorry, but it's wrong format\nTry again")

        else:
            raw = message.photo[-1].file_id
            file_info = bot.get_file(raw)
            downloaded_file = bot.download_file(file_info.file_path)
            image_name = sha256(file_info.file_path.encode('utf-8')).hexdigest() + ".jpg"
            with open(join(config["tg_bot"]["image_path_save"], image_name), 'wb') as new_file:
                new_file.write(downloaded_file)

            attr_a = i.attr_clf.inference(join(config["tg_bot"]["image_path_save"], image_name))
            image_clf_keyboard(message, attr_a[0], i.attrs_d)
            attr_a = attr_a.type(torch.float)
            attr_a = (attr_a * 2 - 1) * 0.5


    @bot.callback_query_handler(func=lambda call: True)
    def callback_worker(call):
        attr_b = attr_a.clone()
        attr_b[0] = resolve_attrs(attr_b[0], i.d_attrs[call.data])

        attr_b[0][i.d_attrs[call.data]] = 1.0 if int(attr_b[0][i.d_attrs[call.data]] != 0.5) else -1.0
        render(attr_b)
        bot.send_photo(call.message.chat.id,
                       photo=open(join(config["tg_bot"]["image_path_save"], "gen_" + image_name), 'rb'))

    def render(attr_b):
        i.image_manipulate(join(config["tg_bot"]["image_path_save"], image_name), attr_a, attr_b)

    bot.polling(none_stop=True, interval=1)

