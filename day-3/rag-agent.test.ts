import { Document } from "@langchain/core/documents";
import { initLogger } from "../lib/logger";
import { checkHallucination } from "./rag-agent.util";

async function testHallucinationCheck() {
  const testCases = [
    {
      name: "정확한 정보",
      documents: [new Document({ pageContent: "사과는 빨간색 과일입니다." })],
      answer: "사과는 빨간색 과일입니다.",
      expectedHallucination: false,
    },
    {
      name: "완전히 잘못된 정보",
      documents: [new Document({ pageContent: "사과는 빨간색 과일입니다." })],
      answer: "사과는 파란색 과일입니다.",
      expectedHallucination: true,
    },
    {
      name: "부분적으로 정확하지만 과장된 정보",
      documents: [new Document({ pageContent: "일부 사과는 빨간색입니다." })],
      answer: "모든 사과는 항상 빨간색입니다.",
      expectedHallucination: true,
    },
    {
      name: "문맥상 맞지 않는 정보",
      documents: [
        new Document({ pageContent: "사과는 과일의 한 종류입니다." }),
      ],
      answer: "사과는 우주선의 연료로 사용됩니다.",
      expectedHallucination: true,
    },
    {
      name: "문서에 없는 정보",
      documents: [new Document({ pageContent: "사과는 빨간색 과일입니다." })],
      answer: "사과는 비타민 C가 풍부합니다.",
      expectedHallucination: true,
    },
  ];

  for (const testCase of testCases) {
    console.log(`테스트 케이스: ${testCase.name}`);
    const isHallucination = await checkHallucination(
      testCase.answer,
      testCase.documents,
    );
    const testPassed = isHallucination === testCase.expectedHallucination;
    console.log(`결과: ${testPassed ? "통과" : "실패"}`);
    console.log(
      `예상: ${testCase.expectedHallucination}, 실제: ${isHallucination}`,
    );
    console.log("\n---\n");
  }
}

async function main() {
  initLogger("./day-3/rag-agent.test.log");
  console.log("환각 체크 테스트 시작");
  console.log("\n---\n");
  await testHallucinationCheck();
  console.log("환각 체크 테스트 완료");
}

main().catch(console.error);
