# `make publish` (or just `make`) will build the archive file based on the contents of this directory.
# The archive will then be published to s3

ARCHIVE=default_dashboard.tar.gz
MODEL_NAME=default_dashboard_model

publish: create
	${MAKE} publish-version VERSION=0.0.1
	${MAKE} publish-version VERSION=latest
	${MAKE} clean

clean:
	rm -f $(ARCHIVE)

create: clean
	# The "-s '/./$(MODEL_NAME)/'" is a sed-like command that prefixes each file with the MODEL_NAME.
	# This is done so that when it's unzipped, it can be treated like the other ootb monitors, which themselves unzip into separate directories per model
	COPYFILE_DISABLE=true tar -s '/./$(MODEL_NAME)/' --exclude='sampleData' --exclude='.git*' --exclude='Makefile' --exclude=$(ARCHIVE) -czvf $(ARCHIVE) .

publish-version:
	aws s3 cp --acl public-read $(ARCHIVE) s3://modelop-download/$(MODEL_NAME)/releases/$(VERSION)/$(ARCHIVE)
