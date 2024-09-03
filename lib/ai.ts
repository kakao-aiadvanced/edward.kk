import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export async function chunkAndEmbed(text: string) {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunks = await textSplitter.splitText(text);

  const embeddings = new OpenAIEmbeddings();
  return { chunks, embeddings };
}

export async function storeInVectorDB(
  chunks: string[],
  embeddings: OpenAIEmbeddings,
) {
  return await MemoryVectorStore.fromTexts(chunks, [], embeddings);
}

export async function buildRAGChain(vectorStore: MemoryVectorStore) {
  const retriever = vectorStore.asRetriever(3);
  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

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
