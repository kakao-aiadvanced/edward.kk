import { ChatOpenAI } from "@langchain/openai";
import { initLogger } from "../lib/logger";

initLogger("./rag-agent.log");

async function main() {
  console.log("RAG Agent 시작");

  // OpenAI 모델 초기화
  const model = new ChatOpenAI({ modelName: "gpt-4o-mini", temperature: 0 });

  // 사용자 질문 (나중에 입력으로 받을 수 있도록 수정 예정)
  const question = "RAG에 대한 저자의 생각은 무엇인가?";

  console.log(`질문: ${question}`);

  // 여기에 라우팅 로직 추가 예정

  // 임시 응답 (나중에 실제 RAG 로직으로 대체 예정)
  const response = await model.predict(
    "이 질문에 대한 답변을 생성해주세요: " + question,
  );

  console.log(`답변: ${response}`);
}

main().catch(console.error);
