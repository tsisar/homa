# Multi-arch build for the Homa backend (HTTP server mode; afplay is only used
# by the local CLI, never in the container).
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t ghcr.io/tsisar/homa:dev --push .
FROM --platform=$BUILDPLATFORM golang:1.25 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG TARGETOS TARGETARCH VERSION=dev
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w -X main.version=${VERSION}" -o /out/homa .

# distroless/static ships CA certificates (HTTPS to Lemonade / MCP) and runs as
# nonroot — a few MB, no shell.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/homa /homa
ENV ADDR=:8080
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/homa"]
