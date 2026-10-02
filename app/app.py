from flask import Flask, render_template, request

from src.recommendation import preload_data
from src.query_parser import recommend_details
from src.posters import add_posters


app = Flask(
    __name__,
    template_folder="view",
    static_folder="public",
    static_url_path="/public"
)

preload_data()


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    error = None
    query = ""

    if request.method == "POST":
        query = request.form["query"].strip()

        if not query:
            error = "Please type what you want to watch."
        else:
            recommendations = add_posters(recommend_details(query, n=16))
            if not recommendations:
                recommendations = None
                error = "No results. Try a popular movie title, e.g. 'Inception but funny'."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error,
        query=query
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080, use_reloader=False)