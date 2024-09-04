import { Document } from "@langchain/core/documents";
import { initLogger } from "../lib/logger";
import { askQuestion } from "../lib/utils";
import {
  buildRAGChain,
  checkHallucination,
  createVectorStore,
  getSourceInfo,
  gradeDocuments,
  regenerateAnswer,
  routeQuestion,
  webSearch,
} from "./rag-agent.util";

async function main() {
  // 로거 초기화
  initLogger("./day-3/rag-agent.log");
  console.log("\n---\n");

  // 사용자 질문
  const question = await askQuestion("질문을 입력하세요: ");
  console.log(`질문: ${question}\n`);

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

  // 라우팅 결정
  const indexTopics = "RAG, 검색 증강 생성, fine-tuning";
  const routingDecision = await routeQuestion(question, indexTopics);
  console.log(`라우팅 결정: ${routingDecision}`);

  let answer: string = "";
  let relevantDocs: Document[] = [];

  if (routingDecision === "vectorstore") {
    // RAG를 사용하여 답변 생성
    const result = await ragChain.invoke({ input: question });
    const initialDocs = result.context;

    // 문서 평가
    relevantDocs = await gradeDocuments(question, initialDocs);

    if (relevantDocs.length) {
      const answerResult = await ragChain.invoke({
        input: question,
        context: relevantDocs,
      });
      answer = answerResult.answer;
    } else {
      console.log("관련 문서가 없습니다. 웹 검색으로 전환합니다.");
    }
  }

  if (!answer) {
    // 웹 검색 로직
    console.log("웹 검색을 수행합니다.");
    relevantDocs = await webSearch(question);

    if (relevantDocs.length) {
      const answerResult = await ragChain.invoke({
        input: question,
        context: relevantDocs,
      });
      answer = answerResult.answer;
      console.log("웹 검색에 성공했습니다.");
    } else {
      answer = "\n최종 답변: 웹 검색 결과를 찾을 수 없습니다.";
      console.log("\n---");
      return;
    }
  }

  // 환각 체크
  let isHallucination = await checkHallucination(answer, relevantDocs);
  let regenerationAttempts = 0;
  const maxRegenerationAttempts = 3;

  while (isHallucination && regenerationAttempts < maxRegenerationAttempts) {
    console.log(
      `환각이 감지되었습니다. 답변을 재생성합니다. (시도 ${regenerationAttempts + 1}/${maxRegenerationAttempts})`,
    );
    console.log("기존 답변:", answer);

    answer = await regenerateAnswer(question, answer, relevantDocs, ragChain);
    isHallucination = await checkHallucination(answer, relevantDocs);
    regenerationAttempts++;
  }

  if (isHallucination) {
    console.log(
      "최대 재생성 시도 횟수를 초과했습니다. 가장 최근 생성된 답변을 사용합니다.",
    );
  }

  console.log(`\n최종 답변: ${answer}`);
  console.log(getSourceInfo(relevantDocs));
  console.log("\n---");
}

main()
  .catch(console.error)
  .finally(async () => {
    setTimeout(() => process.exit(0), 1000);
  });
