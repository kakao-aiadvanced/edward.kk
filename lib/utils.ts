import {
  CheerioWebBaseLoader,
  WebBaseLoaderParams,
} from "@langchain/community/document_loaders/web/cheerio";

export async function scrapeWebsite(
  url: string,
  fields?: WebBaseLoaderParams,
): Promise<string> {
  const loader = new CheerioWebBaseLoader(url, fields);

  const docs = await loader.load();
  return docs.map((doc) => doc.pageContent).join("");
}
