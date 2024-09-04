import { Document } from "@langchain/core/documents";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export async function createVectorStore(
  texts: string[],
): Promise<MemoryVectorStore> {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await textSplitter.createDocuments(texts);

  const embeddings = new OpenAIEmbeddings();
  return await MemoryVectorStore.fromDocuments(docs, embeddings);
}

export async function searchDocuments(
  vectorStore: MemoryVectorStore,
  query: string,
): Promise<Document[]> {
  return await vectorStore.similaritySearch(query, 3);
}

export async function buildRAGChain(vectorStore: MemoryVectorStore) {
  const retriever = vectorStore.asRetriever(3);
  const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });

  const systemTemplate = [
    `You are an assistant for question-answering tasks. `,
    `Use the following pieces of retrieved context to answer `,
    `the question. If you don't know the answer, say that you `,
    `don't know. Use three sentences maximum and keep the `,
    `answer concise.`,
    `\n\n`,
    `{context}`,
  ].join("");

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    ["human", "{input}"],
  ]);

  const questionAnswerChain = await createStuffDocumentsChain({ llm, prompt });
  const chain = createRetrievalChain({
    retriever,
    combineDocsChain: questionAnswerChain,
  });
  return chain;
}

export async function routeQuestion(
  question: string,
  indexTopics: string,
): Promise<string> {
  const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });

  const routingPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert at routing a user question to a vectorstore or web search.
      Use the vectorstore for questions on the following topics: ${indexTopics}.
      You do not need to be stringent with the keywords in the question related to these topics.
      Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
      Return the choice as a JSON with a single key 'datasource' and no preamble or explanation.`,
    ],
    ["human", "{question}"],
  ]);

  const routingChain = routingPrompt
    .pipe(llm)
    .pipe(new JsonOutputParser<{ datasource: "web_search" | "vectorstore" }>());
  const result = await routingChain.invoke({ question });

  return result.datasource;
}

export async function checkHallucination(
  answer: string,
  documents: Document[],
): Promise<boolean> {
  const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });

  const hallucinationCheckPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
      Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by the given documents. 
      Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.`,
    ],
    ["human", `Documents: {documents}\n\nAnswer: {answer}`],
  ]);

  const hallucinationCheckChain = hallucinationCheckPrompt
    .pipe(llm)
    .pipe(new JsonOutputParser<{ score: "yes" | "no" }>());

  const result = await hallucinationCheckChain.invoke({
    documents: documents.map((doc) => doc.pageContent).join("\n"),
    answer,
  });

  return result.score.toLowerCase() === "no";
}

export async function gradeDocuments(
  question: string,
  documents: Document[],
): Promise<Document[]> {
  const llm = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });

  const gradePrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert at assessing the relevance of documents to a given question. 
      Evaluate if the document is relevant to answering the question.
      Give a binary 'yes' or 'no' score to indicate whether the document is relevant.
      Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.`,
    ],
    ["human", `Question: {question}\n\nDocument: {document}`],
  ]);

  const gradeChain = gradePrompt
    .pipe(llm)
    .pipe(new JsonOutputParser<{ score: "yes" | "no" }>());

  const relevantDocs: Document[] = [];

  for (const doc of documents) {
    const result = await gradeChain.invoke({
      question,
      document: doc.pageContent,
    });

    if (result.score.toLowerCase() === "yes") {
      relevantDocs.push(doc);
    }
  }

  return relevantDocs;
}
