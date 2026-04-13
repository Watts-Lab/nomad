import { NextRequest, NextResponse } from "next/server"
import { spawn } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

export const runtime = "nodejs"

type ComparePayload = {
  seed?: number
  mode?: "single" | "compare"
  left: {
    algo: string
    params: Record<string, number>
    cmap?: string
  }
  right: {
    algo: string
    params: Record<string, number>
    cmap?: string
  }
}

function runPythonCompare(payload: ComparePayload): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "scripts", "algorithm_compare.py")
    const preferredPython = "/opt/anaconda3/bin/python"
    const pythonBin =
      process.env.PYTHON_BIN ||
      process.env.PYTHON ||
      (fs.existsSync(preferredPython) ? preferredPython : "python3")

    const proc = spawn(pythonBin, [scriptPath], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        MPLCONFIGDIR: process.env.MPLCONFIGDIR || "/tmp/matplotlib-cache",
      },
    })

    let stdout = ""
    let stderr = ""

    proc.stdout.on("data", (chunk) => {
      stdout += chunk.toString()
    })

    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString()
    })

    proc.on("error", (err) => {
      reject(new Error(`Failed to start python process: ${err.message}`))
    })

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python exited with code ${code}. ${stderr}`))
        return
      }
      try {
        resolve(JSON.parse(stdout))
      } catch (err) {
        reject(new Error(`Invalid JSON from python. ${String(err)}. stderr: ${stderr}`))
      }
    })

    proc.stdin.write(JSON.stringify(payload))
    proc.stdin.end()
  })
}

export async function POST(req: NextRequest) {
  try {
    const payload = (await req.json()) as ComparePayload
    const data = await runPythonCompare(payload)
    return NextResponse.json(data)
  } catch (err) {
    return NextResponse.json(
      {
        error: err instanceof Error ? err.message : "Unknown compare error",
      },
      { status: 500 },
    )
  }
}
