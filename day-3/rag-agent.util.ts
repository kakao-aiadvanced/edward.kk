import { TavilySearchAPIRetriever } from "@langchain/community/retrievers/tavily_search_api";
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

export async function webSearch(query: string): Promise<Document[]> {
  const retriever = new TavilySearchAPIRetriever({
    apiKey: process.env.TAVILY_API_KEY,
    k: 3,
  });

  try {
    const results = await retriever.invoke(query);
    return results;
  } catch (error) {
    console.error("웹 검색 중 오류 발생:", error);
    return [];
  }
}

export function getSourceInfo(docs: Document[]): string {
  if (docs.length === 0) return "출처 없음";

  const source =
    docs[0].metadata.source || docs[0].metadata.url || "알 수 없는 출처";
  return `출처: ${source}`;
}

export async function regenerateAnswer(
  question: string,
  previousAnswer: string,
  relevantDocs: Document[],
  ragChain: any,
): Promise<string> {
  const prompt = `
  원래 질문: ${question}
  이전 답변: ${previousAnswer}
  
  위의 답변에서 환각이 감지되었습니다. 제공된 문서 내용만을 바탕으로 더욱 신중하고 정확한 답변을 생성해주세요.
  확실하지 않은 정보는 포함하지 말고, 문서에 없는 내용은 추측하지 마세요.
  문서에 관련 정보가 없다면 "제공된 정보만으로는 정확한 답변을 할 수 없습니다."라고 대답해주세요.
  `;

  const result = await ragChain.invoke({
    input: prompt,
    context: relevantDocs,
  });

  return result.answer;
}
