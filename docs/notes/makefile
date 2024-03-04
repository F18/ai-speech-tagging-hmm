FILENAME = notes
LATEX = xelatex

all: watch

# Watching for file changes requires the inotify-tools package.
# $ sudo apt-get install inotify-tools
watch:
	while true; do \
		inotifywait -e modify $(FILENAME).md; \
		make pandoc; \
	done

pandoc:
	pandoc $(FILENAME).md -o $(FILENAME).pdf --pdf-engine=$(LATEX)

