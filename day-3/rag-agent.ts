import { Document } from "@langchain/core/documents";
import { initLogger } from "../lib/logger";
import {
  buildRAGChain,
  checkHallucination,
  createVectorStore,
  routeQuestion,
} from "./rag-agent.util";

async function main() {
  // 로거 초기화
  initLogger("./day-3/rag-agent.log");
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
