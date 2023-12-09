function sendMessage() {
    var userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    var messageSection = document.getElementById('message-section');

    // Add user message
    var userMessageContainer = document.createElement('div');
    userMessageContainer.className = 'message user-message';

    // Create the profile icon for the user
    var profileIcon = document.createElement('img');
    profileIcon.src = '/static/user_icon.png';  // Ganti dengan path yang sesuai
    profileIcon.alt = 'User Icon';
    profileIcon.className = 'profile-icon-user';

    // Create the message content
    var messageContent = document.createElement('span');
    messageContent.id = 'user-response';
    messageContent.innerHTML = userInput;

    // Append the profile icon and message content to the user message container
    userMessageContainer.appendChild(profileIcon);
    userMessageContainer.appendChild(messageContent);

    // Append the user message container to the message section
    messageSection.appendChild(userMessageContainer);

    // Call the function to get and display the chatbot response
    getChatbotResponse(userInput);

    // Clear the user input
    document.getElementById('user-input').value = '';

    // Scroll to the bottom to show the latest message
    messageSection.scrollTop = messageSection.scrollHeight;
}

function getChatbotResponse(userInput) {
    var messageSection = document.getElementById('message-section');

    // Make an AJAX request to your Flask server to get the bot response
    fetch('/get_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'user_input=' + encodeURIComponent(userInput),
    })
    .then(response => response.json())
    .then(data => {
        // Inside this block, you can use the actual response
        var messageContainer = document.createElement('div');
        messageContainer.className = 'message bot-message';

        // Create the profile icon
        var profileIcon = document.createElement('img');
        profileIcon.src = '/static/bot_icon.png';
        profileIcon.alt = 'Bot Icon';
        profileIcon.className = 'profile-icon-bot';

        // Create the message content
        var messageContent = document.createElement('span');
        messageContent.innerHTML = 'Chatbot response: ' + data.response;

        // Append the profile icon and message content to the message container
        messageContainer.appendChild(profileIcon);
        messageContainer.appendChild(messageContent);

        // Append the message container to the message section
        messageSection.appendChild(messageContainer);

        // Scroll to the bottom to show the latest message
        messageSection.scrollTop = messageSection.scrollHeight;
    })
    .catch(error => {
        console.error('Error fetching bot response:', error);
        var errorMessageContainer = document.createElement('div');
        errorMessageContainer.className = 'message';
        errorMessageContainer.innerHTML = 'Error fetching response';
        messageSection.appendChild(errorMessageContainer);

        // Scroll to the bottom to show the latest message
        messageSection.scrollTop = messageSection.scrollHeight;
    });
}

function getBotResponseFromServer(userInput) {
    // Make an AJAX request to your Flask server to get the bot response
    // You can use fetch or any other method here
    var response;
    fetch('/get_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'user_input=' + encodeURIComponent(userInput),
    })
    .then(response => response.json())
    .then(data => {
        response = data.response;
    })
    .catch(error => {
        console.error('Error fetching bot response:', error);
        response = 'Error fetching response';
    });

    return response;
}
