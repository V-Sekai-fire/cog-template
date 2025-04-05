predict image:
    cog predict -i input_path=@{{image}} | tee output.json
    jq -r '.prediction_json' output.json | sed 's/data:application\/json;base64,//' | base64 --decode > result.json
    jq -r '.prediction_image' output.json | sed 's/data:image\/png;base64,//' | base64 --decode > prediction_image.png