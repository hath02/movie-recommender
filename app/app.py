from flask import Flask, render_template, request

from src.recommendation import (
    recommend_for_user,
    get_user_rated_movies,
    preload_data
)


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

    movie = ""
    user_input = ""

    if request.method == "POST":

        movie = request.form["movie"].strip()
        user_input = request.form["user"].strip()

        # Check movie input
        if not movie:
            error = "Invalid Movie Title."

            return render_template(
                "index.html",
                recommendations=None,
                error=error,
                movie=movie,
                user=user_input
            )

        # Check user input
        if not user_input:
            error = "Invalid User ID."

            return render_template(
                "index.html",
                recommendations=None,
                error=error,
                movie=movie,
                user=user_input
            )

        # Convert User ID
        try:
            user_id = int(user_input)

        except ValueError:
            error = "User ID must be a number."

            return render_template(
                "index.html",
                recommendations=None,
                error=error,
                movie=movie,
                user=user_input
            )

        # Get rated movies
        rated_movies = get_user_rated_movies(user_id)

        # Generate recommendations
        recommendations = recommend_for_user(
            movie,
            user_id,
            n=16,
            exclude_movies=rated_movies
        )

        # No recommendations
        if recommendations.height == 0:
            error = "No recommendations found."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error,
        movie=movie,
        user=user_input
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        port=8080,
        use_reloader=False
    )