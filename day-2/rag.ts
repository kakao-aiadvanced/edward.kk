import { buildRAGChain, chunkAndEmbed, storeInVectorDB } from "../lib/ai";
import { initLogger } from "../lib/logger";
import { scrapeWebsite } from "../lib/utils";

initLogger("./day-2/rag.log");

async function main() {
  const url = "https://applied-llms.org/";

  console.log("\n✦ 웹 페이지 스크래핑 중...");
  const text = await scrapeWebsite(url, {
    selector: ".content",
  });

  console.log("\n✦ 텍스트 청크화 및 임베딩 중...");
  const { chunks, embeddings } = await chunkAndEmbed(text);
  console.log(`✧ 총 ${chunks.length}개의 청크가 생성되었습니다.`);

  console.log("\n✦ 벡터 데이터베이스에 저장 중...");
  const vectorStore = await storeInVectorDB(chunks, embeddings);
  console.log(
    `✧ 총 ${vectorStore.memoryVectors.length}개의 벡터가 생성되었습니다.`,
  );

  console.log("\n✦ RAG 체인 구축 중...");
  const ragChain = await buildRAGChain(vectorStore);
  console.log("\n✧ RAG 체인 구축 완료");

  const questions = [
    "RAG에 대한 저자의 생각은 무엇인가?",
    "RAG와 fine tuning에 대해 저자는 어떻게 비교하고 있나?",
    "저자가 가장 많은 부분을 할당해 설명하는 개념은 무엇인가?",
  ];

  for (const question of questions) {
    console.log(`\n질문: ${question}`);
    const res = await ragChain.invoke({ input: question });
    console.log(`답변: ${res.answer}`);
  }
}

main().catch(console.error);
