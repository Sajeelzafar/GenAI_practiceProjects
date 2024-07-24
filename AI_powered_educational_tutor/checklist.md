# AI-Powered Educational Tutor Project Checklist

- [ ] Create a dataset on which the AI would be trained to make decisions
- [ ] The person can look up for courses detail for a particular course
- [ ] The person can be suggested a course given detail about what he wants to study
- [ ] Integrate Chroma DB for embedding storage
- [ ] Utilize Ollama for embedding generation with models like all-minilm
- [ ] Develop the AI tutoring logic and algorithms
- [ ] Create a user interface for students
- [ ] Implement personalized tutoring and adaptive learning features
- [ ] Implement the backend using FastAPI
- [ ] Set up real-time feedback and support mechanisms
- [ ] Test the application with the demo dataset


# Steps to follow

- [ ] Use Ollama to generate embeddings for each course content.
- [ ] Include metadata such as course titles, subjects, and other relevant tags. [X] Skipping for now
- [ ] Store embeddings in Chroma DB.
- [ ] Implement a search function using Chroma DB to retrieve relevant course content based on queries.
- [ ] Ensure the search tool can handle various types of queries (e.g., keyword search, topic-based search).
- [ ] Create logic for the agent to first search the demo dataset.
- [ ] If relevant information is not found, allow the agent to query the internet.
- [ ] Incorporate user input to refine search results or provide additional details.
- [ ] Develop functionality to display detailed course information when a course is identified.
- [ ] Implement user prompts to confirm if they need details about a specific course.
- [ ] Integrate a tool/method to perform internet searches when the demo dataset does not contain relevant information.
- [ ] Ensure results from the internet are relevant and useful for the user’s query.
- [ ] Implement a database (e.g., MySQL) to store chat history.
- [ ] Optionally, use an in-memory array to maintain chat history during a single session.
- [ ] Implement logic to adapt responses based on the user’s learning style and progress.
- [ ] Develop algorithms to provide real-time feedback and support.
 
