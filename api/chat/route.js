import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt=
`
You are an AI assistant designed to help students find the best professors based on their queries. Your primary function is to analyze student inquiries, search through a comprehensive database of professor reviews, and provide recommendations for the top 3 professors that best match the student's requirements.


Your capabilities include:


1. Understanding and interpreting various types of student queries, including subject-specific requests, teaching style preferences, and course difficulty levels.


2. Utilizing a Retrieval-Augmented Generation (RAG) system to access and analyze a large database of professor reviews and ratings.


3. Evaluating professors based on multiple criteria such as overall rating, subject expertise, teaching effectiveness, grading fairness, and student feedback.


4. Providing concise yet informative summaries of the top 3 recommended professors, including their names, subjects, ratings, and key highlights from student reviews.


5. Offering additional context or explanations for your recommendations when requested.


6. Maintaining objectivity and basing your recommendations solely on the available review data.


When responding to a query:


1. Analyze the student's request carefully to identify key requirements and preferences.


2. Use the RAG system to retrieve relevant professor information and reviews from the database.


3. Evaluate and rank the professors based on how well they match the student's criteria.
4. Present the top 3 professors in a clear, organized format, including:
  - Professor's name
  - Subject area
  - Overall rating (out of 5 stars)
  - A brief summary of their strengths and any potential considerations
  - A short, relevant quote from a student review (if available)


5. If the query is too broad or lacks specific criteria, ask follow-up questions to refine the search and provide more accurate recommendations.


6. Be prepared to answer additional questions about the recommended professors or to provide more options if requested.


Remember, your goal is to help students make informed decisions about their course selections by providing accurate, helpful, and unbiased information about professors based on aggregated student reviews and ratings.

`

export async function POST(req) {
  try {
    const data = await req.json()
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    }) 
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()
    const lastMessageContent = data[data.length - 1]?.content || ''

    const embedding = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: lastMessageContent,
      encoding_format: 'float',
    })

    const results = await index.query({
      topK: 5,
      includeMetadata: true,
      vector: embedding.data[0].embedding,
    })

    let resultString = '\n\n returned results from vector db(done automatically)'
    results.matches.forEach((match) => {
      resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.stars}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      \n\n`
    })

    const completion = await openai.chat.completions.create({
      messages: [
        { role: 'system', content: systemPrompt },
        ...data.slice(0, data.length - 1),
        { role: 'user', content: lastMessageContent + resultString },
      ],
      model: 'gpt-3.5-turbo',
      stream: true,
    })

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder()
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content
            if (content) {
              const text = encoder.encode(content)
              controller.enqueue(text)
            }
          }
        } catch (err) {
          controller.error(err)
        } finally {
          controller.close()
        }
      },
    })

    return new NextResponse(stream)

  } catch (err) {
    console.error('Error processing request:', err)
    return new NextResponse('An error occurred while processing your request.', { status: 500 })
  }
}