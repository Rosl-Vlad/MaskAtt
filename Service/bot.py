import yaml
import telebot

from os.path import join
from telebot import types
from hashlib import sha256

from src import Implementation

token = open("tg_token.txt", 'r')
bot = telebot.TeleBot(token.read())


def image_clf_keyboard(message, attrs, attrs_name):
    keyboard = types.InlineKeyboardMarkup()
    for i, at in enumerate(attrs):
        key = types.InlineKeyboardButton(text=attrs_name[i] + " - " + str(int(attrs[i].item())),
                                         callback_data=attrs_name[i])
        keyboard.add(key)

    question = 'Какой атрибут меняем?'
    bot.send_message(message.from_user.id, text=question, reply_markup=keyboard)


if __name__ == "__main__":
    f = open("config.yaml", 'r')
    config = yaml.safe_load(f)
    f.close()

    i = Implementation(config)

    print("Model ready!")


    @bot.message_handler(content_types=['photo'])
    def get_text_messages(message):
        global attr_a
        global image_name

        raw = message.photo[-1].file_id
        file_info = bot.get_file(raw)
        downloaded_file = bot.download_file(file_info.file_path)
        image_name = sha256(file_info.file_path.encode('utf-8')).hexdigest() + ".jpg"
        with open(join(config["tg_bot"]["image_path_save"], image_name), 'wb') as new_file:
            new_file.write(downloaded_file)

        attr_a = i.attr_clf.inference(join(config["tg_bot"]["image_path_save"], image_name))

        image_clf_keyboard(message, attr_a[0], i.attrs_d)


    @bot.callback_query_handler(func=lambda call: True)
    def callback_worker(call):
        attr_b = attr_a.clone()

        attr_b[0][i.d_attrs[call.data]] = int(attr_b[0][i.d_attrs[call.data]] != 1)

        i.image_manipulate(join(config["tg_bot"]["image_path_save"], image_name), attr_a, attr_b)

        bot.send_photo(call.message.chat.id, photo=open(join(config["tg_bot"]["image_path_save"], "gen_" + image_name), 'rb'))

    bot.polling(none_stop=True, interval=1)
