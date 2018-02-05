var debug = process.env.NODE_ENV !== "production";
var webpack = require('webpack');
var path = require('path');

debug = false;

module.exports = {
  context: path.join(__dirname, "src"),
  devtool: debug ? "inline-sourcemap" : null,
  entry: "./index.js",
  module: {
    loaders: [
      {
        test: /\.jsx?$/,
        exclude: /(node_modules|bower_components)/,
        loader: 'babel-loader',
        query: {
          presets: ['react', 'es2015','stage-0'],
          plugins: ['transform-decorators-legacy']
        }
      },{
        test: /\.scss$/,
        loaders: ['style', 'css', 'sass']
      }
    ]
  },
  output: {
    path: __dirname + '/lib/',
    filename: "App.min.js"
  },
  plugins: debug ? [] : [
    new webpack.DefinePlugin({
      'process.env': {
        'NODE_ENV': '"production"'
      }
    }),
    new webpack.optimize.DedupePlugin(),
    new webpack.optimize.OccurenceOrderPlugin(),
    new webpack.optimize.UglifyJsPlugin({
      mangle: true, 
      sourcemap: false, 
      compress: {
        warnings: false,
      },
      output: {
        comments: false
      },
      minimize: true,
    }),
  ],
};