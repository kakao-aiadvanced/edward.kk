import {
  CheerioWebBaseLoader,
  WebBaseLoaderParams,
} from "@langchain/community/document_loaders/web/cheerio";
import readline from "readline";

export async function scrapeWebsite(
  url: string,
  fields?: WebBaseLoaderParams,
): Promise<string> {
  const loader = new CheerioWebBaseLoader(url, fields);

  const docs = await loader.load();
  return docs.map((doc) => doc.pageContent).join("");
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

export async function askQuestion(query: string) {
  return new Promise<string>((resolve) => {
    rl.question(query, (answer) => {
      resolve(answer);
    });
  });
}
