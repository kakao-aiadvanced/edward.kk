import fs from "fs";
import util from "util";

export function initLogger(filepath: string) {
  // 로그 파일 스트림 생성
  const logFile = fs.createWriteStream(filepath, { flags: "a" });

  // 콘솔 출력을 파일로 리다이렉트
  const originalConsoleLog = console.log;
  console.log = function (...args) {
    originalConsoleLog.apply(console, args);
    logFile.write(util.format.apply(null, args) + "\n");
  };

  // 에러 출력도 파일로 리다이렉트
  const originalConsoleError = console.error;
  console.error = function (...args) {
    originalConsoleError.apply(console, args);
    logFile.write("ERROR: " + util.format.apply(null, args) + "\n");
  };

  // 프로그램 종료 시 로그 파일 닫기
  process.on("exit", () => {
    logFile.end();
  });
}
