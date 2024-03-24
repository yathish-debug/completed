css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
}

.chat-message.user {
    background-color: #2b313e;
    color: #fff;
}

.chat-message.bot {
    background-color: #475063;
    color: #fff;
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
}
</style>
'''

# Define your bot and user templates here
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <a href='https://postimages.org/' target='_blank'><img src='https://i.postimg.cc/m1c0WbcG/bot.png' border='0' alt='bot'/></a>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <a href='https://postimages.org/' target='_blank'><img src='https://i.postimg.cc/bZf44fpK/woman.png' border='0' alt='woman'/></a>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''