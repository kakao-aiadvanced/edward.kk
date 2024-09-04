import { Document } from "@langchain/core/documents";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
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

async function main() {
  console.log("RAG Agent 시작");

  // OpenAI 모델 초기화
  const model = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });

  // 벡터 저장소 생성 (실제 데이터로 대체 필요)
  const sampleTexts = [
    "RAG는 검색 증강 생성의 약자로, 외부 지식을 활용하여 언어 모델의 응답을 개선하는 기술입니다.",
    "저자는 RAG가 fine-tuning에 비해 더 유연하고 업데이트가 쉽다고 주장합니다.",
    "RAG의 주요 장점은 최신 정보를 쉽게 통합할 수 있다는 것입니다.",
  ];
  const vectorStore = await createVectorStore(sampleTexts);
  console.log("벡터 저장소 생성 완료");

  // 사용자 질문
  const question = "RAG에 대한 저자의 생각은 무엇인가?";
  console.log(`질문: ${question}`);

  // 문서 검색
  const relevantDocs = await searchDocuments(vectorStore, question);
  console.log(
    "관련 문서 검색 완료:",
    relevantDocs.map((doc) => doc.pageContent),
  );

  // 여기에 검색 결과를 바탕으로 답변 생성 로직 추가 예정

  // 임시 응답 (나중에 실제 RAG 로직으로 대체 예정)
  const response = await model.predict(
    "이 질문에 대한 답변을 생성해주세요: " + question,
  );
  console.log(`답변: ${response}`);
}

main().catch(console.error);
