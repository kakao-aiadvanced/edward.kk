import { Document } from "@langchain/core/documents";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { initLogger } from "../lib/logger";

// 로거 초기화
initLogger("./day-3/rag-agent.log");

async function createVectorStore(texts: string[]): Promise<MemoryVectorStore> {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await textSplitter.createDocuments(texts);

  const embeddings = new OpenAIEmbeddings();
  return await MemoryVectorStore.fromDocuments(docs, embeddings);
}

async function searchDocuments(
  vectorStore: MemoryVectorStore,
  query: string,
): Promise<Document[]> {
  return await vectorStore.similaritySearch(query, 3);
}

async function buildRAGChain(vectorStore: MemoryVectorStore) {
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

async function routeQuestion(
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

async function checkHallucination(
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

async function main() {
  console.log("\n---\n\n");

  // 벡터 저장소 생성 (실제 데이터로 대체 필요)
  const sampleTexts = [
    "RAG는 검색 증강 생성의 약자로, 외부 지식을 활용하여 언어 모델의 응답을 개선하는 기술입니다.",
    "저자는 RAG가 fine-tuning에 비해 더 유연하고 업데이트가 쉽다고 주장합니다.",
    "RAG의 주요 장점은 최신 정보를 쉽게 통합할 수 있다는 것입니다.",
  ];
  const vectorStore = await createVectorStore(sampleTexts);
  console.log("벡터 저장소 생성 완료");

  // RAG 체인 구축
  const ragChain = await buildRAGChain(vectorStore);
  console.log("RAG 체인 구축 완료");

  // 사용자 질문
  const question = "RAG에 대한 저자의 생각은 무엇인가?";
  console.log(`질문: ${question}`);

  // 라우팅 결정
  const indexTopics = "RAG, 검색 증강 생성, fine-tuning";
  const routingDecision = await routeQuestion(question, indexTopics);
  console.log(`라우팅 결정: ${routingDecision}`);

  let answer: string;
  let relevantDocs: Document[];

  if (routingDecision === "vectorstore") {
    // RAG를 사용하여 답변 생성
    const result = await ragChain.invoke({ input: question });
    answer = result.answer;
    relevantDocs = result.context;
  } else {
    // 웹 검색 로직 (아직 구현되지 않음)
    answer = "웹 검색 기능은 아직 구현되지 않았습니다.";
    relevantDocs = [];
  }

  // 환각 체크
  const isHallucination = await checkHallucination(answer, relevantDocs);
  if (isHallucination) {
    console.log("환각이 감지되었습니다. 답변을 재생성합니다.");
    // 여기에 답변 재생성 로직 추가
    answer =
      "환각이 감지되어 답변을 재생성해야 합니다. (재생성 로직은 아직 구현되지 않았습니다.)";
  }

  console.log(`최종 답변: ${answer}`);

  console.log("\n---");
}

main().catch(console.error);
