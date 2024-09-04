import { Document } from "@langchain/core/documents";
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

async function main() {
  console.log("\n---\n\nRAG Agent 시작");

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

  // RAG를 사용하여 답변 생성
  const result = await ragChain.invoke({ input: question });
  console.log(`답변: ${result.answer}`);

  console.log("\n---");
}

main().catch(console.error);
